from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from app.prompts import AFTER_SALE_EXTRACTION_PROMPT
from app.schemas import AfterSaleExtraction


class AfterSaleService:
    def __init__(self, model: Any):
        self.model = (
            model.with_structured_output(
                AfterSaleExtraction,
                method="function_calling",
            )
            if hasattr(model, "with_structured_output")
            else model
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", AFTER_SALE_EXTRACTION_PROMPT),
                ("human", "{text}"),
            ]
        )

    def extract(self, text: str) -> AfterSaleExtraction:
        messages = self.prompt.invoke({"text": text}).to_messages()
        result = self.model.invoke(messages)
        return AfterSaleExtraction.model_validate(result)
