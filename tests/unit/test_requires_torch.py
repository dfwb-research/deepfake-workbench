def test_requires_torch_skips_or_imports(requires_torch):
    torch = requires_torch
    assert torch.__name__ == "torch"
