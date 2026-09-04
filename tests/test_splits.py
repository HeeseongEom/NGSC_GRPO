from ngsc_grpo.splits import stratified_take


def test_stratified_take_is_deterministic_and_balanced():
    records = [
        {"image_id": f"a{i}", "present_classes": ["a"]} for i in range(8)
    ] + [{"image_id": f"b{i}", "present_classes": ["b"]} for i in range(8)]
    first, _ = stratified_take(records, 8, 2027)
    second, _ = stratified_take(records, 8, 2027)
    assert [row["image_id"] for row in first] == [row["image_id"] for row in second]
    assert sum(row["present_classes"] == ["a"] for row in first) == 4
