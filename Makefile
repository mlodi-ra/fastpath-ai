.PHONY: install demo evaluate test benchmark build

install:
	python -m pip install -e ".[dev,train]"

demo:
	python -m examples.ticket_triage.generate_data
	python -m examples.ticket_triage.train
	python -m fastpath evaluate --model examples/ticket_triage/fastpath-ticket-triage-v0.json --schema examples.ticket_triage.schema:InboundTicketTriage --state "Production DB CPU at 99 percent and connections exhausted"

evaluate:
	python -m examples.ticket_triage.evaluate

test:
	pytest

benchmark:
	python -m benchmarks.latency

build:
	python -m build
