def _header(title: str) -> None:
    print("\n" + "=" * 65)
    print(f" {title}")
    print("=" * 65)


def _sub(title: str) -> None:
    print(f"\n--- {title} ---")


def _slug(name: str) -> str:
    return (name.replace(" ", "_").replace("(", "").replace(")", "")
                .replace("/", "").replace("Δ", "D").replace(".", ""))
