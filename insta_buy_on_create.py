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

from solders.keypair import Keypair
from solders.rpc.config import RpcSendTransactionConfig
from solders.rpc.responses import SendTransactionResp
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts
from solana.rpc.commitment import Processed

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

# Pump.fun program id
PUMP_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# -------- UTILS --------
def now_ms() -> str:
    t = time.time()
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1)*1000):03d}"

def b58_keypair_from_secret(secret_b58: str) -> Keypair:
    sk = base58.b58decode(secret_b58)
    return Keypair.from_bytes(sk)

def parse_create_instruction(data: bytes) -> Optional[Dict[str, Any]]:
    """Same structure as your listener; kept here for self-containment."""
    if len(data) < 8:
        return None
    offset = 8
    parsed_data = {}
    fields = [
        ("name", "string"),
        ("symbol", "string"),
        ("uri", "string"),
        ("mint", "publicKey"),
        ("bondingCurve", "publicKey"),
        ("user", "publicKey"),
        ("creator", "publicKey"),
    ]
    try:
        for field_name, field_type in fields:
            if field_type == "string":
                length = struct.unpack("<I", data[offset: offset + 4])[0]
                offset += 4
                value = data[offset: offset + length].decode("utf-8")
                offset += length
            elif field_type == "publicKey":
                value = base58.b58encode(data[offset: offset + 32]).decode("utf-8")
                offset += 32
            parsed_data[field_name] = value
        return parsed_data
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
        self._latest_blockhash = None
        self._alive = True

    async def start(self):
        # Background task to refresh recent blockhash (~ every 300ms)
        asyncio.create_task(self._refresh_blockhash_loop())

    async def close(self):
        self._alive = False
        await self.client.close()

    async def _refresh_blockhash_loop(self):
        while self._alive:
            try:
                res = await self.client.get_latest_blockhash(Processed)
                if res.value:
                    self._latest_blockhash = res.value.blockhash
            except Exception:
                pass
            await asyncio.sleep(0.3)

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
        raw = _tx_to_bytes(tx)  # VersionedTransaction is bytes-like
        try:
            resp = await self.client.send_raw_transaction(
                raw,
                opts=TxOpts(skip_preflight=True, preflight_commitment=Processed, max_retries=0)
            )
            sig = str(resp.value) if hasattr(resp, "value") else str(resp)
            print(f"[{now_ms()}] [BUY→SENT] {sig}  mint={ev.mint}  {sol_amount} SOL")
            return sig
        except Exception as e:
            print(f"[{now_ms()}] [BUY→ERROR] {e!s}")
            return None

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
                close_timeout=1
            ) as ws:
                await ws.send(json.dumps(sub))
                _ = await ws.recv()  # subscription ack

                print(f"[{now_ms()}] [create] listening @ {WSS_ENDPOINT} (processed)")
                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    if msg.get("method") != "logsNotification":
                        continue

                    value = msg["params"]["result"]["value"]
                    logs = value.get("logs", [])

                    # Very fast pre-filter
                    fast_hit = False
                    for line in logs:
                        if "Program log: Instruction: Create" in line:
                            fast_hit = True
                            break
                    if not fast_hit:
                        continue

                    # Find base64 "Program data:"
                    encoded = None
                    for line in logs:
                        if "Program data:" in line:
                            encoded = line.split(": ", 1)[1]
                            break
                    if not encoded:
                        continue

                    try:
                        decoded = base64.b64decode(encoded)
                        parsed = parse_create_instruction(decoded)
                        if not parsed or "mint" not in parsed or "bondingCurve" not in parsed:
                            continue

                        ev = CreateEvent.from_parsed(value.get("signature", ""), parsed)
                        # Log the detection & fire buy immediately
                        print(f"[{now_ms()}] [CREATE] sig={ev.signature[:8]}… mint={ev.mint} bc={ev.bonding_curve}")
                        asyncio.create_task(engine.buy_on_create(ev, SOL_TO_SPEND, SLIPPAGE_BPS))
                    except Exception as e:
                        # Keep going even if one decode fails
                        print(f"[{now_ms()}] [decode_err] {e!s}")
        except Exception as e:
            print(f"[{now_ms()}] [ws_err] {e!s}  (reconnect in 2s)")
            await asyncio.sleep(2.0)

if __name__ == "__main__":
    asyncio.run(listen_and_buy())
