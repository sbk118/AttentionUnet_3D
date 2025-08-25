import hashlib

#비밀번호 해싱 함수
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

#사용자 정보 (비밀번호는 반드시 해시!)
users = {
    "kdt001": hash_password("oracle123"),
    "kdt002": hash_password("oracle456")
}

#로그인 검증 함수
def verify_user(username: str, password: str) -> bool:
    hashed = hash_password(password)
    return users.get(username) == hashed