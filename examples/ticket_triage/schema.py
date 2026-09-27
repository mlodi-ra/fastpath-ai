from pydantic import BaseModel

from fastpath import Choice, Noul, Score


class InboundTicketTriage(BaseModel):
    category: Choice["Infrastructure", "Billing", "Security", "General"]  # noqa: F821
    urgency: Score[1, 5]
    requires_tier3_page: Noul
