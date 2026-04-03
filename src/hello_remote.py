from pathlib import Path


def add_numbers(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    total = add_numbers(2, 3)
    print(f"Hello from the remote server. 2 + 3 = {total}")

    notes_file = Path("student_notes.txt")
    if notes_file.exists():
        print("Found student_notes.txt. First line:")
        with notes_file.open() as f:
            print(f.readline().strip())
    else:
        print("student_notes.txt not found yet. Create it as part of the tutorial.")
