def _custom_pretty_formatter(item, **_kwargs) -> str:
    class_name = item.__class__.__name__

    field_values = {}
    if hasattr(item, "__dataclass_fields__"):
        for field_name_df in getattr(item, "__dataclass_fields__", {}).keys():
            try:
                field_values[field_name_df] = getattr(item, field_name_df)
            except AttributeError:
                pass
    elif hasattr(item, "__dict__"):
        field_values = item.__dict__
    else:
        return repr(item)

    if not field_values:
        return f"{class_name}()"

    parts = []
    for name, value in field_values.items():
        try:
            value_str = str(value)
        except Exception:
            value_str = "<error converting value to string>"

        if "\n" in value_str:
            indented_value = "\n".join(["    " + line for line in value_str.split("\n")])
            parts.append(f"  {name}: \n{indented_value}")
        else:
            parts.append(f"  {name}: {value_str}")

    return f"{class_name}(\n" + ",\n".join(parts) + "\n)"
