# pumpfun_tx_builder.py  (solders-only)
import os, base64, json
import httpx

from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders.message import to_bytes_versioned
from solders.pubkey import Pubkey

QN_PUMPFUN_SWAP_URL = os.getenv("QN_PUMPFUN_SWAP_URL")

async def build_buy_tx(
    client,                 # unused here
    payer: Keypair,         # solders.Keypair
    mint: str,
    bonding_curve: str,     # not needed by Metis; it derives it
    lamports: int,
    slippage_bps: int,
    cu_price_micro_lamports: int,
    compute_unit_limit: int,
    recent_blockhash=None,
):
    if not QN_PUMPFUN_SWAP_URL:
        raise RuntimeError("Set QN_PUMPFUN_SWAP_URL to your QuickNode /pump-fun/swap endpoint.")

    # --- inside pumpfun_tx_builder.py ---
    payload = {
        "wallet": str(payer.pubkey()),
        "type": "BUY",
        "mint": mint,
        "inAmount": str(lamports),
        # keep it minimal first — add these back later if desired:
        # "priorityFeeLevel": "high",
        # "slippageBps": str(slippage_bps),
        # "commitment": "processed",
    }

    async with httpx.AsyncClient(timeout=2.0) as http:
        r = await http.post(QN_PUMPFUN_SWAP_URL, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(
                f"Metis /pump-fun/swap error {r.status_code}: {r.text[:500]}"
            )
        body = r.json()
        tx_b64 = body.get("tx") or body.get("transaction")
        if not tx_b64:
            raise RuntimeError(f"Pump.fun swap response missing tx field: {body}")


    raw = base64.b64decode(tx_b64)

    # Deserialize → sign message → replace our signature slot → return tx
    vtx = VersionedTransaction.from_bytes(raw)
    msg = vtx.message
    msg_bytes = to_bytes_versioned(msg)

    # Find our signer index among account_keys
    keys = list(msg.account_keys)
    try:
        my_idx = next(i for i, k in enumerate(keys) if k == payer.pubkey())
    except StopIteration:
        raise RuntimeError("Payer pubkey not found in transaction account keys; cannot sign.")

    sigs = list(vtx.signatures)  # list of solders.signature.Signature
    sigs[my_idx] = payer.sign_message(msg_bytes)
    vtx.signatures = sigs  # mutate signatures per solders docs

    return vtx  # orchestrator will bytes(...) this and send
