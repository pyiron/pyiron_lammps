import tomlkit


if __name__ == "__main__":
    with open("pyproject.toml", "r") as f:
        data = tomlkit.load(f)

    lst = []
    if "optional-dependencies" in data["project"]:
        for sub_lst in data["project"]["optional-dependencies"].values():
            for el in sub_lst:
                lst.append(el)

    data["project"]["dependencies"] += [
        el for el in set(lst) if not el.startswith("pwtools")
    ]

    with open("pyproject.toml", "w") as f:
        f.writelines(tomlkit.dumps(data))
