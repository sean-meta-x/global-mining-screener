"""
Manage whitelist users for Mining Screener.
Usage:
    python manage_users.py add <username> <password> [display_name]
    python manage_users.py remove <username>
    python manage_users.py list
    python manage_users.py reset <username> <new_password>
"""
import sys
import hashlib
import os

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

AUTH_FILE = os.path.join(os.path.dirname(__file__), "auth_users.yaml")


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load() -> dict:
    if not os.path.exists(AUTH_FILE):
        return {"users": {}}
    with open(AUTH_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"users": {}}


def _save(data: dict) -> None:
    with open(AUTH_FILE, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def cmd_add(username: str, password: str, display_name: str = "") -> None:
    data = _load()
    if username in data["users"]:
        print(f"User '{username}' already exists. Use 'reset' to change password.")
        return
    data["users"][username] = {
        "password_hash": _hash(password),
        "display_name": display_name or username,
    }
    _save(data)
    print(f"Added user: {username} (display: {display_name or username})")


def cmd_remove(username: str) -> None:
    data = _load()
    if username not in data["users"]:
        print(f"User '{username}' not found.")
        return
    del data["users"][username]
    _save(data)
    print(f"Removed user: {username}")


def cmd_reset(username: str, new_password: str) -> None:
    data = _load()
    if username not in data["users"]:
        print(f"User '{username}' not found. Use 'add' to create.")
        return
    data["users"][username]["password_hash"] = _hash(new_password)
    _save(data)
    print(f"Password updated for: {username}")


def cmd_list() -> None:
    data = _load()
    users = data.get("users", {})
    if not users:
        print("No users configured (open access).")
        return
    print(f"{'Username':<20} {'Display Name'}")
    print("-" * 40)
    for uname, info in users.items():
        print(f"{uname:<20} {info.get('display_name', uname)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0].lower()
    if cmd == "add" and len(args) >= 3:
        cmd_add(args[1], args[2], args[3] if len(args) > 3 else "")
    elif cmd == "remove" and len(args) == 2:
        cmd_remove(args[1])
    elif cmd == "reset" and len(args) == 3:
        cmd_reset(args[1], args[2])
    elif cmd == "list":
        cmd_list()
    else:
        print(__doc__)
