import hashlib
from os import environ
from pathlib import Path

# import onnxruntime
from Crypto import Random
from Crypto.Cipher import AES

StrPath = str | Path

MODEL_ENCRYPTION_SECURE_KEY = environ.get("MODEL_ENCRYPTION_SECURE_KEY")
assert MODEL_ENCRYPTION_SECURE_KEY != "", "MODEL_ENCRYPTION_SECURE_KEY must be provided"


def load_graph(path: StrPath) -> bytes:
    path = Path(path)

    with path.open("rb") as f:
        protobuf_byte_str = f.read()
    return protobuf_byte_str


def encrypt_file(raw: bytes, _key: str) -> bytes:
    bs = 32
    key = hashlib.sha256(_key.encode()).digest()
    s = raw
    raw = s + str.encode((bs - len(s) % bs) * chr(bs - len(s) % bs))
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return iv + cipher.encrypt(raw)


def decrypt_file(enc: bytes, _key: str) -> bytes:
    key = hashlib.sha256(_key.encode()).digest()
    iv = enc[: AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    s = cipher.decrypt(enc[AES.block_size :])
    return s[: -ord(s[len(s) - 1 :])]


def encrypt_model(
    input_path: StrPath,
    output_path: StrPath,
    enc_key: StrPath = MODEL_ENCRYPTION_SECURE_KEY,
):
    output_path = Path(output_path)

    # encode
    nodes_binary_str = load_graph(input_path)
    nodes_binary_str = encrypt_file(nodes_binary_str, enc_key)
    with output_path.open("wb") as f:
        f.write(nodes_binary_str)

    return output_path


def decrypt_model(
    model_path: StrPath,
    output_path: StrPath,
    enc_key: str = MODEL_ENCRYPTION_SECURE_KEY,
):
    model_path = Path(model_path)
    output_path = Path(output_path)

    with model_path.open("rb") as f:
        enc = f.read()

    decoded = decrypt_file(enc, enc_key)

    with output_path.open("wb") as f:
        f.write(decoded)
