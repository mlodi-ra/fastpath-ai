import pytest
from pydantic import BaseModel

from fastpath import Choice, Noul, Score
from fastpath.errors import ContractError
from fastpath.schema import compile_schema


class ValidContract(BaseModel):
    route: Choice["A", "B", "C"]  # noqa: F821
    risk: Score[0, 100]
    block: Noul


def test_compiles_all_head_types():
    contract = compile_schema(ValidContract)
    assert [head.kind for head in contract.heads] == ["choice", "score", "noul"]
    assert contract.heads[0].options == ("A", "B", "C")
    assert len(contract.fingerprint) == 16


def test_fingerprint_is_deterministic():
    assert compile_schema(ValidContract).fingerprint == compile_schema(ValidContract).fingerprint


def test_rejects_ordinary_fields():
    class InvalidContract(BaseModel):
        free_text: str

    with pytest.raises(ContractError):
        compile_schema(InvalidContract)


def test_rejects_invalid_choice_and_score():
    with pytest.raises(TypeError):
        Choice["only"]
    with pytest.raises(TypeError):
        Score[5, 1]
