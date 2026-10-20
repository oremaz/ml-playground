from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from ml_pipeline import benchmark_agent as agent
from ml_pipeline.utils.playground_trace import append_event, export_history, history_markdown, operation, read_events


def client_response(content, **message_fields):
    return SimpleNamespace(id="response-1", model="actual-model", usage=SimpleNamespace(
        prompt_tokens=20, completion_tokens=10, total_tokens=30), choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=content, tool_calls=None, **message_fields))])


def mock_client(monkeypatch, response):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    monkeypatch.setattr(agent, "create_client", lambda *args, **kwargs: client)


def test_logs_full_messages_provider_output_usage_without_credentials(tmp_path, monkeypatch):
    response = client_response('{"analysis": "Observed result"}', reasoning_content="hidden provider internals")
    mock_client(monkeypatch, response)
    settings = {"provider": "openrouter", "model": "requested-model", "api_key": "private-provider-key"}
    result = agent.complete_json(settings, "System instructions", {"context": "User context"},
                                 workspace=tmp_path, stage="interpretation")
    assert result["analysis"] == "Observed result"
    request, completion = read_events(tmp_path)
    assert request["operation_id"] == completion["operation_id"]
    assert request["data"]["messages"][0]["content"] == "System instructions"
    assert json.loads(request["data"]["messages"][1]["content"])["context"] == "User context"
    assert completion["data"]["model"] == "actual-model"
    assert completion["data"]["usage"]["total_tokens"] == 30
    assert completion["data"]["message"]["content"] == response.choices[0].message.content
    raw = (tmp_path / "events.jsonl").read_text()
    assert "private-provider-key" not in raw
    assert "hidden provider internals" not in raw


def test_bad_json_keeps_response_and_records_failure(tmp_path, monkeypatch):
    mock_client(monkeypatch, client_response("incomplete JSON"))
    with pytest.raises(ValueError):
        agent.complete_json({"provider": "local", "model": "test"}, "prompt", {}, workspace=tmp_path)
    events = read_events(tmp_path)
    assert [e["event"] for e in events] == ["llm.request", "llm.response", "llm.error"]
    assert events[1]["data"]["message"]["content"] == "incomplete JSON"
    assert events[2]["data"]["error_type"] == "JSONDecodeError"


@pytest.mark.parametrize("reasoning,explicit,expected", [(None, None, 3072), ("none", None, 3072),
                                                       ("low", None, 8192), ("low", 1024, 1024),
                                                       ("xhigh", None, 16384)])
def test_native_reasoning_has_output_budget_and_preserves_explicit_limit(tmp_path, monkeypatch, reasoning, explicit, expected):
    received = []
    def create(**kwargs):
        received.append(kwargs)
        return client_response("Measured results only.")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(agent, "create_client", lambda *a, **k: client)
    result = agent.complete_message({"provider": "local", "model": "test", "reasoning": reasoning},
                                    [{"role": "user", "content": "Compare actual results"}], [],
                                    workspace=tmp_path, max_tokens=explicit)
    assert received[0]["max_tokens"] == expected
    assert read_events(tmp_path)[0]["data"]["max_tokens"] == expected
    assert result["content"] == "Measured results only."
    if reasoning == "low":
        assert received[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_interpretation_records_actual_result_payload_and_reply(tmp_path, monkeypatch):
    workspace = agent.create_workspace(tmp_path, [("data.csv", b"x,target\n1,0\n2,1\n")])
    agent.write_json(workspace / "plan.json", {"target": "target"})
    result = {"output": "runs/original", "dataset_summary": {"validation_rows": 2}}
    agent.write_json(workspace / "last_result.json", result)
    mock_client(monkeypatch, client_response('{"analysis": "Two validation rows"}'))
    agent.interpret_results(workspace, {"provider": "local", "model": "test"})
    assert agent.read_json(workspace / "conversation.json")[-1]["content"] == "Two validation rows"
    request = next(event for event in read_events(workspace) if event["event"] == "llm.request")
    assert request["operation"] == "interpretation"
    assert json.loads(request["data"]["messages"][1]["content"])["result"] == result


def test_provider_error_redacts_credentials(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Rejected private-provider-key")
    monkeypatch.setattr(agent, "create_client", fail)
    with pytest.raises(RuntimeError):
        agent.complete_json({"provider": "groq", "model": "test", "api_key": "private-provider-key"},
                            "prompt", {}, workspace=tmp_path)
    raw = (tmp_path / "events.jsonl").read_text()
    assert "private-provider-key" not in raw
    assert read_events(tmp_path)[-1]["data"]["error"] == "Rejected [REDACTED]"


def test_native_tool_requests_are_recorded_but_not_dispatched(tmp_path, monkeypatch):
    response = client_response("{}")
    response.choices[0].message.tool_calls = [{"id": "call-1", "type": "function",
        "function": {"name": "some_tool", "arguments": "{}"}}]
    mock_client(monkeypatch, response)
    with pytest.raises(ValueError, match="no native tool dispatcher"):
        agent.complete_json({"provider": "local", "model": "test"}, "prompt", {}, workspace=tmp_path)
    events = read_events(tmp_path)
    assert events[1]["data"]["message"]["tool_calls"][0]["function"]["name"] == "some_tool"
    assert not any(e["event"].startswith("execution.") for e in events)


def test_concurrent_events_remain_valid_json_lines(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda index: append_event(tmp_path, "test", data={"index": index, "text": "x" * 5000}), range(40)))
    events = read_events(tmp_path)
    assert sorted(e["data"]["index"] for e in events) == list(range(40))
    assert len({e["event_id"] for e in events}) == 40


def test_operation_failure_records_duration_and_preserves_exception(tmp_path):
    with pytest.raises(ValueError, match="bad join"):
        with operation(tmp_path, "data.prepare", {"table": "data.csv"}, parent_id="parent", run_id="run"):
            raise ValueError("bad join")
    start, error = read_events(tmp_path)
    assert start["operation_id"] == error["operation_id"]
    assert error["event"] == "execution.error"
    assert error["parent_id"] == "parent"
    assert error["run_id"] == "run"
    assert error["data"]["elapsed_seconds"] >= 0


def test_legacy_exports_preserve_original_artifacts_without_inventing_events(tmp_path):
    agent.write_json(tmp_path / "conversation.json", [{"role": "user", "content": "Original objective"},
                                                   {"role": "assistant", "content": {"insights": ["Original insight"]}}])
    (tmp_path / "revision_0001.py").write_text("print('original script')\n")
    (tmp_path / "run_original.log").write_text("Original stdout\n")
    history = export_history(tmp_path)
    assert history["events"] == []
    assert "cannot be recovered" in history["note"]
    rendered = history_markdown(history)
    assert "Original objective" in rendered
    assert "Original insight" in rendered
    assert "original script" in rendered
    assert "Original stdout" in rendered
    assert not (tmp_path / "events.jsonl").exists()


def test_wide_context_compaction_preserves_protocol_scores_labels_and_original_log(tmp_path, monkeypatch):
    columns = [f'sensor_{index}' for index in range(561)]
    profile = {'rows': 100, 'column_count': 562, 'column_names': ['target', *columns],
               'columns': [{'name': key, 'statistics': {'mean': 1}} for key in ['target', *columns[:59]]],
               'sample': [{key: 1 for key in columns[:59]}] * 2}
    fact = {'id': 't1.example.none.custom_recall', 'value': .92, 'description': 'Observed sitting recall'}
    messages = [
        {'role': 'user', 'content': json.dumps({'reviewed_config': {'features': columns}})},
        {'role': 'assistant', 'tool_calls': [{'id': 'inspect', 'function': {'name': 'inspect_dataset'}}]},
        {'role': 'tool', 'tool_call_id': 'inspect', 'content': json.dumps({'profiles': {'development': profile}})},
        {'role': 'assistant', 'tool_calls': [{'id': 'metric', 'function': {'name': 'evaluate_metric'}}]},
        {'role': 'tool', 'tool_call_id': 'metric', 'content': json.dumps({'definition': {'name': 'custom_recall'},
            'by_trial': [{'trial': 1, 'results': [{'value': .92, 'class_label': 'SITTING'}],
                          'evidence': [fact, {'id': 't1.other', 'value': 1}]}]})},
        {'role': 'assistant', 'content': 'Preserve research notes. ' * 4000}]
    original = json.loads(json.dumps(messages))
    compact = agent.compact_agent_messages(messages)
    assert messages == original

    assert compact[0] == messages[0] and compact[-1] == messages[-1]
    assert [(m['role'], m.get('tool_call_id')) for m in compact] == [(m['role'], m.get('tool_call_id')) for m in messages]
    result = json.loads(compact[2]['content'])['profiles']['development']
    assert result['rows'] == 100 and result['column_count'] == 562
    assert len(result['columns']) == 16 and len(result['sample']) == 1
    result = json.loads(compact[4]['content'])
    assert result['definition'] == {'name': 'custom_recall'}
    assert result['by_trial'][0]['results'] == [{'value': .92, 'class_label': 'SITTING'}]
    assert result['by_trial'][0]['evidence'] == [fact]
    assert len(json.dumps(compact)) < len(json.dumps(messages))
    received = []
    def create(**kwargs):
        received.append(kwargs)
        return client_response("Ready for the next measured comparison.")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(agent, "create_client", lambda *a, **k: client)
    agent.complete_message({"provider": "local", "model": "test"}, messages, [], workspace=tmp_path)
    request = read_events(tmp_path)[0]["data"]
    assert request["messages"] == received[0]["messages"] == compact
    assert request["original_message_characters"] == len(json.dumps(original))
    assert messages == original


def test_state_memory_preserves_dynamic_evidence_and_exact_incumbent_without_nested_history(tmp_path, monkeypatch):
    columns = [f'sensor_{index}' for index in range(561)]
    subset = columns[:272]
    trial = {'number': 1, 'status': 'completed', 'config': {'features': subset, 'models': ['linear_classifier']},
             'recipes': [], 'summary': [{'model': 'linear_classifier', 'augmentation': 'none', 'f1_macro': .95}],
             'diagnostics': {'diagnostics': [{'model': 'linear_classifier', 'augmentation': 'none',
                 'per_class': [{'label': 'SITTING', 'precision': .91, 'recall': .92, 'f1': .915, 'rows': 20}]}]}}
    state = {'trials': [trial], 'config': {'metric': 'f1_macro', 'features': columns},
             'approved_models': ['linear_classifier'], 'approval_requested': True, 'max_trials': 5,
             'best': {'trial': 1, 'model': 'linear_classifier', 'f1_macro': .95}, 'notes': []}
    messages = [{'role': 'system', 'content': 'Follow the experiment protocol'},
                {'role': 'user', 'content': json.dumps({'reviewed_config': {'features': columns}})},
                {'role': 'tool', 'content': json.dumps({'nested_history': [columns] * 100})}]
    messages.extend([{'role': 'assistant', 'tool_calls': [{'id': 'rejected', 'function': {
        'name': 'evaluate_metric', 'arguments': json.dumps({'definition_json': json.dumps({'name': 'sitting_recall'})})}}]},
        {'role': 'tool', 'tool_call_id': 'rejected', 'content': json.dumps({'error': 'Metric names require custom_'})}])
    original = json.loads(json.dumps(messages))
    original_state = json.loads(json.dumps(state))
    agent.write_json(tmp_path / 'state.json', state)
    received = []
    def create(**kwargs):
        received.append(kwargs)
        return client_response('Review the measured recall before the next experiment.')
    monkeypatch.setattr(agent, 'create_client', lambda *a, **k: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    agent.complete_message({'provider': 'local', 'model': 'test'}, messages, [], workspace=tmp_path)
    projected = received[0]['messages']
    memory = json.loads(projected[1]['content'].split('Persisted experiment memory:\n')[1])
    assert [message['role'] for message in projected] == ['system', 'user']
    assert memory['incumbent_exact_features'] == subset
    assert memory['approved_models'] == ['linear_classifier'] and memory['trials_remaining'] == 4
    assert memory['label_order'] == ['SITTING']
    assert memory['recent_tool_errors'] == [{'tool': 'evaluate_metric',
        'rejected_arguments': {'definition_json': {'name': 'sitting_recall'}},
        'result': {'error': 'Metric names require custom_'}}]
    assert memory['trials'][0]['evidence_values']['t1.linear_classifier.none.class.0.recall'] == .92
    assert 'nested_history' not in projected[1]['content']
    assert len(json.dumps(projected)) < len(json.dumps(messages)) / 5
    assert read_events(tmp_path)[0]['data']['messages'] == projected
    assert messages == original and state == original_state
