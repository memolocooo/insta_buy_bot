import os
import json
import asyncio
import base64
import struct
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any
import base58
import websockets
from dotenv import load_dotenv
import httpx
from solders.keypair import Keypair
from solders.rpc.config import RpcSendTransactionConfig
from solders.rpc.responses import SendTransactionResp
from solders.signature import Signature
from solders.transaction import VersionedTransaction
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts
from solana.rpc.commitment import Processed

try:
    import orjson as json  # pip install orjson
except Exception:
    import json


# -------- ENV --------
load_dotenv()
WSS_ENDPOINT = os.getenv("SOLANA_NODE_WSS_ENDPOINT", "wss://mainnet.helius-rpc.com/?api-key=YOUR_KEY")
RPC_HTTP = os.getenv("SOLANA_NODE_RPC_HTTP_ENDPOINT", "https://mainnet.helius-rpc.com/?api-key=YOUR_KEY")
PRIVATE_KEY = os.getenv("TRADER_PRIVATE_KEY_B58")  # base58-encoded 64-byte secret key (NOT the 12/24 seed)
SOL_TO_SPEND = float(os.getenv("BUY_SOL_AMOUNT", "0.05"))
SLIPPAGE_BPS = int(os.getenv("SLIPPAGE_BPS", "100"))  # 1.00%
CU_PRICE_MICROLAMPORTS = int(os.getenv("CU_PRICE_MICROLAMPORTS", "2500"))  # priority fee per CU
COMPUTE_UNIT_LIMIT = int(os.getenv("COMPUTE_UNIT_LIMIT", "600000"))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
JITO_URL = os.getenv("JITO_URL", "https://ny.mainnet.block-engine.jito.wtf/api/v1/transactions")
JITO_AUTH = os.getenv("JITO_AUTH")  # optional UUID token; not required for default rate limits

HOLDERS_WINDOW_MS = int(os.getenv("HOLDERS_WINDOW_MS", "60000"))
HOLDERS_POLL_MS   = int(os.getenv("HOLDERS_POLL_MS", "120"))


# Pump.fun program id
PUMP_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# -------- UTILS --------
def now_ms() -> str:
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1)*1000):03d}"

VERBOSE = os.getenv("VERBOSE", "0") == "1"
def dbg(msg: str):
    if VERBOSE:
        print(msg)

def b58_keypair_from_secret(secret_b58: str) -> Keypair:
    sk = base58.b58decode(secret_b58)
    return Keypair.from_bytes(sk)

def parse_create_instruction(data: bytes) -> Optional[Dict[str, Any]]:
    """
    Minimal parser: skip string decoding (name/symbol/uri), only return
    mint, bondingCurve, creator as base58. Much lighter on the hot path.
    """
    try:
        if len(data) < 8:
            return None
        o = 8  # skip 8-byte method discriminator

        # skip name (string)
        if o + 4 > len(data): return None
        L = struct.unpack_from("<I", data, o)[0]; o += 4 + L

        # skip symbol (string)
        if o + 4 > len(data): return None
        L = struct.unpack_from("<I", data, o)[0]; o += 4 + L

        # skip uri (string)
        if o + 4 > len(data): return None
        L = struct.unpack_from("<I", data, o)[0]; o += 4 + L

        # mint (32), bondingCurve (32), user (32), creator (32)
        if o + 32*4 > len(data): return None
        mint_b58   = base58.b58encode(data[o:o+32]).decode("ascii"); o += 32
        bc_b58     = base58.b58encode(data[o:o+32]).decode("ascii"); o += 32
        o += 32  # skip user
        creator_b58= base58.b58encode(data[o:o+32]).decode("ascii"); o += 32

        return {"mint": mint_b58, "bondingCurve": bc_b58, "creator": creator_b58}
    except Exception:
        return None


def _tx_to_bytes(tx) -> bytes:
    # solana-py VersionedTransaction has .serialize()
    if hasattr(tx, "serialize"):
        return tx.serialize()
    # solders VersionedTransaction supports bytes()
    try:
        return bytes(tx)
    except Exception:
        pass
    # already bytes/bytearray?
    if isinstance(tx, (bytes, bytearray)):
        return bytes(tx)
    raise RuntimeError("Unsupported transaction object; cannot serialize to bytes.")


async def send_via_jito(base64_tx: str) -> str:
    """POST to Jito block engine; returns signature on success."""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sendTransaction",
        "params": [base64_tx, {"encoding": "base64"}]
    }
    headers = {"Content-Type": "application/json"}
    if JITO_AUTH:
        headers["x-jito-auth"] = JITO_AUTH

    async with httpx.AsyncClient(timeout=2.0) as http:
        r = await http.post(JITO_URL, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"Jito error: {data['error']}")
        sig = data.get("result") or data.get("value")
        if not sig:
            raise RuntimeError(f"Jito unexpected response: {data}")
        return str(sig)



def _tx_to_bytes(tx) -> bytes:
    if hasattr(tx, "serialize"):
        return tx.serialize()  # solana-py VersionedTransaction
    try:
        return bytes(tx)       # solders VersionedTransaction
    except Exception:
        pass
    if isinstance(tx, (bytes, bytearray)):
        return bytes(tx)
    raise RuntimeError("Unsupported transaction object; cannot serialize to bytes.")


@dataclass
class CreateEvent:
    signature: str
    name: str
    symbol: str
    uri: str
    mint: str
    bonding_curve: str
    creator: str

    @classmethod
    def from_parsed(cls, sig: str, d: Dict[str, Any]) -> "CreateEvent":
        return cls(
            signature=sig,
            name=d.get("name", ""),
            symbol=d.get("symbol", ""),
            uri=d.get("uri", ""),
            mint=d.get("mint", ""),
            bonding_curve=d.get("bondingCurve", ""),
            creator=d.get("creator", ""),
        )

# -------- BUY ENGINE --------
class BuyEngine:
    def __init__(self, rpc_http: str, keypair: Keypair):
        self.rpc_http = rpc_http
        self.kp = keypair
        self.client = AsyncClient(self.rpc_http, timeout=5)
        self._http = httpx.AsyncClient(timeout=2.0, headers={"Content-Type": "application/json"})
        self._latest_blockhash = None
        self._alive = True

    async def close(self):
        self._alive = False
        await self.client.close()
        await self._http.aclose()


    async def start(self):
        # Background task to refresh recent blockhash (~ every 300ms)
        asyncio.create_task(self._refresh_blockhash_loop())


    async def _refresh_blockhash_loop(self):
        while self._alive:
            try:
                res = await self.client.get_latest_blockhash(Processed)
                if res.value:
                    self._latest_blockhash = res.value.blockhash  # keep as solders.Hash
            except Exception:
                pass
            await asyncio.sleep(0.15)  # was 0.3 — tighter refresh

    
    async def send_via_rpc(self, raw_tx: bytes) -> str:
        resp = await self.client.send_raw_transaction(
            raw_tx,
            opts=TxOpts(skip_preflight=True, preflight_commitment=Processed, max_retries=0)
        )
        sig = getattr(resp, "value", None) or getattr(resp, "result", None) or resp
        return str(sig)

    async def send_via_jito(self, base64_tx: str) -> str:
        body = {
            "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
            "params": [base64_tx, {"encoding": "base64"}]
        }
        headers = {}
        if JITO_AUTH:
            headers["x-jito-auth"] = JITO_AUTH
        r = await self._http.post(JITO_URL, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"Jito error: {data['error']}")
        return str(data.get("result") or data.get("value"))



    async def buy_on_create(self, ev: CreateEvent, sol_amount: float, slippage_bps: int) -> Optional[str]:
        lamports = int(sol_amount * 1_000_000_000)
        if DRY_RUN:
            print(f"[{now_ms()}] [DRY] Would BUY {ev.mint} (bc={ev.bonding_curve}) for {sol_amount} SOL")
            return f"dry_{ev.mint[:6]}"

        # IMPORTANT: You likely already have a Pump.fun tx builder (e.g., build_pumpfun_buy_tx).
        # Hook it up here for zero extra HTTP. Keep priority compute budget.
        try:
            from pumpfun_tx_builder import build_buy_tx  # YOU provide this
            tx: VersionedTransaction = await build_buy_tx(
                client=self.client,
                payer=self.kp,
                mint=ev.mint,
                bonding_curve=ev.bonding_curve,
                lamports=lamports,
                slippage_bps=slippage_bps,
                cu_price_micro_lamports=CU_PRICE_MICROLAMPORTS,
                compute_unit_limit=COMPUTE_UNIT_LIMIT,
                recent_blockhash=self._latest_blockhash,
            )
        except ImportError:
            # If you don't have a builder yet, raise a clear hint.
            raise RuntimeError(
                "Missing `pumpfun_tx_builder.build_buy_tx`.\n"
                "Create it to directly craft the Pump.fun Buy instruction with:\n"
                "  - payer (Keypair)\n"
                "  - mint, bonding_curve\n"
                "  - lamports (amount in SOL)\n"
                "  - slippage_bps, CU price & CU limit\n"
                "  - recent_blockhash (optional)\n"
                "Return a signed solders VersionedTransaction."
            )

        # Fire-and-forget fast send; skip preflight for speed.
                # Fire-and-forget fast send; skip preflight for speed.
        t0 = time.perf_counter()

        raw = _tx_to_bytes(tx)
        b64 = base64.b64encode(raw).decode("ascii")

        rpc_task  = asyncio.create_task(self.send_via_rpc(raw))
        jito_task = asyncio.create_task(self.send_via_jito(b64))

        done, pending = await asyncio.wait({rpc_task, jito_task}, return_when=asyncio.FIRST_COMPLETED)

        winner = None
        sig = None
        for t in done:
            try:
                sig = await t
                winner = "RPC" if t is rpc_task else "JITO"
                break
            except Exception:
                pass

        if sig is None:
            # await the other if first finisher errored
            remaining = list(pending)
            if remaining:
                try:
                    sig = await remaining[0]
                    winner = "RPC" if remaining[0] is rpc_task else "JITO"
                except Exception as e:
                    print(f"[{now_ms()}] [BUY→ERROR] rpc+jito both failed: {e!s}")
                    return None

        # Cancel the slower path
        for p in pending:
            p.cancel()

        dt_ms = int((time.perf_counter() - t0) * 1000)
        print(f"[{now_ms()}] [BUY→SENT][{winner}] {sig}  mint={ev.mint}  {sol_amount} SOL  ({dt_ms}ms send)")
        return sig



# -------- LISTENER --------
async def listen_and_buy():
    if not PRIVATE_KEY:
        raise RuntimeError("Set TRADER_PRIVATE_KEY_B58 in your .env (base58 secret key).")

    kp = b58_keypair_from_secret(PRIVATE_KEY)
    engine = BuyEngine(RPC_HTTP, kp)
    await engine.start()

    # Minimal logsSubscribe payload
    sub = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "logsSubscribe",
        "params": [
            {"mentions": [PUMP_PROGRAM_ID]},
            {"commitment": "processed"},
        ],
    }

    while True:
        try:
            async with websockets.connect(
                WSS_ENDPOINT,
                ping_interval=15,
                ping_timeout=10,
                max_size=None,
                close_timeout=1,
                compression=None,    # <—
                max_queue=1
            ) as ws:
                await ws.send(json.dumps(sub))
                _ = await ws.recv()  # subscription ack

                print(f"[{now_ms()}] [create] listening @ {WSS_ENDPOINT} (processed)")
                while True:
                    raw = await ws.recv()
                    # Pre-filter cheap string patterns to skip JSON parsing for irrelevant messages
                    s = raw if isinstance(raw, str) else raw.decode("utf-8", "ignore")
                    if '"logsNotification"' not in s or 'Program log: Instruction: Create' not in s:
                        continue

                    # Only now do the heavier JSON parse
                    msg = json.loads(s)


                    value = msg["params"]["result"]["value"]
                    logs = value.get("logs", [])

                    # We already pre-filtered the raw message; just grab the encoded data
                    encoded = next((ln.split(": ", 1)[1] for ln in logs if ln.startswith("Program data: ")), None)
                    if not encoded:
                        continue


                    # ... after you've found `encoded` ...

                    try:
                        # faster base64 (no strict validate)
                        decoded = base64.b64decode(encoded, validate=False)

                        parsed = parse_create_instruction(decoded)
                        if not parsed or "mint" not in parsed or "bondingCurve" not in parsed:
                            continue

                        ev = CreateEvent.from_parsed(value.get("signature", ""), parsed)
                        print(f"[{now_ms()}] [CREATE] sig={ev.signature[:8]}… mint={ev.mint} bc={ev.bonding_curve}")
                        asyncio.create_task(engine.buy_on_create(ev, SOL_TO_SPEND, SLIPPAGE_BPS))

                    except Exception as e:
                        # Use debug gate; avoid noisy prints on the hot path
                        dbg(f"[{now_ms()}] [decode_err] {e!s}")
                        continue

        except Exception as e:
            print(f"[{now_ms()}] [ws_err] {e!s}  (reconnect in 2s)")
            await asyncio.sleep(2.0)

if __name__ == "__main__":
    asyncio.run(listen_and_buy())
