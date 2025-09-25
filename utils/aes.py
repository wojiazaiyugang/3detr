"""
AES加解密 依赖pycryptodome库
"""
import base64
import json
from typing import Dict

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

# 随机生成的秘钥
key = b')\xc0\xe6p}\x17\xd1A)\x19-\x88\xc6\tU\xfa\xd6\xc17*\xe6%\xe3>\xb4\x8d&\xf6F\xce\x18~'

def encrypt(data: str) -> Dict[str, str]:
    """
    AES加密
    :param data: 待加密数据
    :return: 加密后的数据，base64编码
    """
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_data = pad(data.encode('utf-8'), AES.block_size)
    encrypted = cipher.encrypt(padded_data)
    return {
        "iv": base64.b64encode(iv).decode('utf-8'),
        "data": base64.b64encode(encrypted).decode('utf-8')
    }

def decrypt(encrypted_data: Dict[str, str]) -> str:
    """
    AES解密
    :param encrypted_data: 加密后的数据，包含iv和ciphertext，均为base64编码
    :return: 解密后的数据
    """
    iv = base64.b64decode(encrypted_data["iv"])
    ciphertext = base64.b64decode(encrypted_data["data"])
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(ciphertext)
    decrypted = unpad(decrypted_padded, AES.block_size)
    return decrypted.decode('utf-8')

if __name__ == '__main__':
    a = json.dumps({"12312": ["wqeqw"]})
    encrypted = encrypt(a)
    print("Encrypted:", encrypted)
    decrypted = json.loads(decrypt(encrypted))
    print("Decrypted:", decrypted)