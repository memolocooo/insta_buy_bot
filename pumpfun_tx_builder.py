# pumpfun_tx_builder.py
# Build a Pump.fun BUY tx LOCALLY with solders (zero HTTP), then sign & return.
import math
from typing import Optional, List

from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.message import MessageV0, Message, to_bytes_versioned
from solders.transaction import VersionedTransaction
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solders.system_program import ID as SYS_PROGRAM_ID

# ---- Constants from on-chain references ----
# Pump.fun program + well-known accounts/ids (documented publicly)
PUMP_FUN_PROGRAM = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
PUMP_GLOBAL_ADDRESS = Pubkey.from_string("4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf")
PUMP_FEE_ADDRESS    = Pubkey.from_string("CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM")
EVENT_AUTHORITY     = Pubkey.from_string("Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1")
TOKEN_PROGRAM_ID    = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
RENT_SYSVAR_ID      = Pubkey.from_string("SysvarRent111111111111111111111111111111111")
ASSOCIATED_TOKEN_PROGRAM_ID = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")

# Pump.fun "buy" method bytes + data layout (8-byte method, then u64 token_amount, u64 lamports)
PUMP_BUY_METHOD = bytes([0x66, 0x06, 0x3d, 0x12, 0x01, 0xda, 0xeb, 0xea])  # from public impl

def _le_u64(n: int) -> bytes:
    return int(n).to_bytes(8, "little", signed=False)

def _derive_ata(owner: Pubkey, mint: Pubkey) -> Pubkey:
    # ATA address = PDA with seeds [owner, TOKEN_PROGRAM_ID, mint] under Associated Token Program
    return Pubkey.find_program_address(
        [bytes(owner), bytes(TOKEN_PROGRAM_ID), bytes(mint)],
        ASSOCIATED_TOKEN_PROGRAM_ID
    )[0]

def _ix_create_ata_idempotent(payer: Pubkey, owner: Pubkey, mint: Pubkey) -> Instruction:
    ata = _derive_ata(owner, mint)
    # Associated Token Program: CreateIdempotent instruction index = 1, no data
    # Accounts (read/write flags per SPL docs): payer(w), ata(w), owner(r), mint(r), sys(r), token(r), rent(r)
    metas = [
        AccountMeta(payer, is_signer=True, is_writable=True),
        AccountMeta(ata,   is_signer=False, is_writable=True),
        AccountMeta(owner, is_signer=False, is_writable=False),
        AccountMeta(mint,  is_signer=False, is_writable=False),
        AccountMeta(SYS_PROGRAM_ID,        is_signer=False, is_writable=False),
        AccountMeta(TOKEN_PROGRAM_ID,      is_signer=False, is_writable=False),
        AccountMeta(RENT_SYSVAR_ID,        is_signer=False, is_writable=False),
    ]
    return Instruction(ASSOCIATED_TOKEN_PROGRAM_ID, bytes([1]), metas)

def _ix_pumpfun_buy(
    payer: Pubkey,
    mint: Pubkey,
    bonding_curve: Pubkey,
    max_lamports: int,
) -> Instruction:
    # Accounts order for Pump.fun buy (per public references):
    # 0  Global (r)
    # 1  Fee Recipient (w)
    # 2  Mint (r)
    # 3  Bonding Curve (w)
    # 4  Associated Bonding Curve (w)
    # 5  Associated User (w)
    # 6  User (payer) (w, signer)
    # 7  System Program (r)
    # 8  Token Program (r)
    # 9  Rent Sysvar (r)
    # 10 Event Authority (r)
    associated_bonding_curve = _derive_ata(bonding_curve, mint)
    associated_user          = _derive_ata(payer, mint)

    metas = [
        AccountMeta(PUMP_GLOBAL_ADDRESS,  False, False),
        AccountMeta(PUMP_FEE_ADDRESS,     False, True),
        AccountMeta(mint,                 False, False),
        AccountMeta(bonding_curve,        False, True),
        AccountMeta(associated_bonding_curve, False, True),
        AccountMeta(associated_user,      False, True),
        AccountMeta(payer,                True,  True),
        AccountMeta(SYS_PROGRAM_ID,       False, False),
        AccountMeta(TOKEN_PROGRAM_ID,     False, False),
        AccountMeta(RENT_SYSVAR_ID,       False, False),
        AccountMeta(EVENT_AUTHORITY,      False, False),
    ]

    # Data: [8 bytes method][8 bytes token_amount][8 bytes lamports]
    # token_amount is 0 for BUY; lamports is max SOL to spend (slippage bound)
    data = PUMP_BUY_METHOD + _le_u64(0) + _le_u64(max_lamports)
    return Instruction(PUMP_FUN_PROGRAM, data, metas)

async def build_buy_tx(
    client,                 # AsyncClient (only used for recent blockhash if not provided)
    payer: Keypair,
    mint: str,
    bonding_curve: str,
    lamports: int,          # desired SOL in lamports (pre-slippage)
    slippage_bps: int,
    cu_price_micro_lamports: int,
    compute_unit_limit: int,
    recent_blockhash: Optional[str] = None,
) -> VersionedTransaction:
    # Slippage guard — max lamports we allow the program to take
    max_lamports = math.floor(lamports * (1 + (slippage_bps / 10_000)))

    mint_pk          = Pubkey.from_string(mint)
    bonding_curve_pk = Pubkey.from_string(bonding_curve)
    payer_pk         = payer.pubkey()

    # Build ixs:
    ixs: List[Instruction] = [
        set_compute_unit_limit(compute_unit_limit),
        set_compute_unit_price(cu_price_micro_lamports),
        _ix_create_ata_idempotent(payer_pk, payer_pk, mint_pk),     # ensure user ATA exists
        _ix_pumpfun_buy(payer_pk, mint_pk, bonding_curve_pk, max_lamports),
    ]

    # Recent blockhash
    from solders.hash import Hash

    # ...
    if recent_blockhash is None:
        # Fallback: fetch one (this returns a solders.Hash)
        res = await client.get_latest_blockhash()
        recent_blockhash = res.value.blockhash  # <-- keep as Hash

    # Accept either a Hash or a str
    if isinstance(recent_blockhash, Hash):
        bh = recent_blockhash
    else:
        bh = Hash.from_string(str(recent_blockhash))


    # Compile & sign v0 message
    msg = MessageV0.try_compile(
        payer_pk,
        ixs,
        [],       # no LUTs
        bh
    )
    tx = VersionedTransaction(msg, [payer])
    return tx
