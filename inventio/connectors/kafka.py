"""Kafka topics: one card per topic (schema.py), from the brokers and a Schema Registry.

    inventio init kafka://broker:9092                                   -> source kafka-broker
    inventio init "kafka://broker:9092?registry=http://registry:8081"

A card holds the partitions, retention and cleanup policy, the value's fields and the key's
type: from the registry's latest schema when the topic has one (subject `<topic>-value`), else
inferred from the newest message when that message is JSON. Only the field names and their
types are kept, never a value: a message is production data. Topics whose name starts with `_`
(Kafka's and the registry's own) are left out. Needs confluent-kafka
(`pip install "inventio[data]"`)."""

import hashlib
import json
import urllib.parse

from . import mirror
from .markdown import segment

KIND = "kafka"
DOC_TYPE = "Dataset"


def origin(url: str, query: str | None = None) -> str | None:
    if not url.startswith("kafka://"):
        return None
    if query:
        raise ValueError("--jql applies to a Jira URL")
    u = urllib.parse.urlsplit(url)
    registry = urllib.parse.parse_qs(u.query).get("registry", [""])[0].rstrip("/")
    return f"kafka://{u.netloc}" + (f"?registry={registry}" if registry else "")


def locate(url: str):
    return None


def heading_url(item: dict, heads: list[str]) -> str | None:
    return None


def fetch_url(url: str):
    return None


class Remote:
    FORMAT = 1  # raise when the Markdown written changes, so every mirror is written again

    def __init__(self, origin_url: str):
        self.origin = origin_url
        u = urllib.parse.urlsplit(origin_url)
        self.brokers = u.netloc
        self.registry = urllib.parse.parse_qs(u.query).get("registry", [""])[0]
        self.name = f"kafka-{self.brokers.split(',')[0].split(':')[0]}"
        self.docs: dict[str, str] = {}

    def listing(self) -> dict[str, mirror.Entry]:
        from .http import RemoteError

        try:
            from confluent_kafka import Consumer, KafkaException, TopicPartition
            from confluent_kafka.admin import AdminClient, ConfigResource
        except ImportError:
            raise RemoteError("reading Kafka topics needs confluent-kafka: pip install \"inventio[data]\"") from None
        conf = {"bootstrap.servers": self.brokers}
        try:
            admin = AdminClient(conf)
            md = admin.list_topics(timeout=15)
            topics = sorted(t for t in md.topics if not t.startswith("_"))
            futures = admin.describe_configs([ConfigResource("topic", t) for t in topics]) if topics else {}
            configs = {res.name: {k: v.value for k, v in f.result().items()} for res, f in futures.items()}
        except KafkaException as e:
            raise RemoteError(f"cannot reach Kafka at {self.brokers}: {e}") from None
        subjects = set(self._registry("/subjects") or []) if self.registry else set()
        consumer = Consumer({**conf, "group.id": "inventio-schema", "enable.auto.commit": False,
                             "enable.partition.eof": False})
        out = {}
        try:
            for t in topics:
                parts = sorted(md.topics[t].partitions)
                # a message is read only when no schema is registered, and only for its field names
                sample = None if f"{t}-value" in subjects else _newest(consumer, TopicPartition, t, parts)
                self.docs[t] = self._card(t, len(parts), configs.get(t, {}), subjects, sample)
                out[t] = mirror.Entry(hashlib.sha1(self.docs[t].encode()).hexdigest(), f"{segment(t)}.md", "")
        finally:
            consumer.close()
        return out

    def fetch(self, ids: list[str]):
        for i in ids:
            yield i, mirror.Doc(self.docs[i])

    def _registry(self, path: str):
        from .http import Client

        return Client(self.registry, {"Accept": "application/vnd.schemaregistry.v1+json"}).get(path)

    def _card(self, topic: str, n_parts: int, cfg: dict, subjects: set, sample) -> str:
        from ..schema import card

        about = [f"Kafka topic on `{self.brokers}`: {n_parts} partition{'s' if n_parts != 1 else ''}, "
                 f"retention {_retention(cfg.get('retention.ms'))}, cleanup `{cfg.get('cleanup.policy', 'delete')}`."]
        rows: list[list[str]] = []
        value = subjects and f"{topic}-value" in subjects and self._registry(f"/subjects/{topic}-value/versions/latest")
        body = _json(sample.value()) if sample is not None else None
        if value:
            kind = value.get("schemaType", "AVRO")
            about.append(f"Value schema: {kind}, registry subject `{topic}-value` version {value['version']}.")
            rows = _schema_fields(kind, value["schema"])
        elif isinstance(body, dict):
            about.append("Value schema: none registered; field names and types inferred from the newest "
                         "message (JSON), its values not kept.")
            rows = [[k, _json_type(v), ""] for k, v in body.items()]
        else:
            about.append("Value schema: none registered" + ("; the topic is empty." if sample is None else
                                                            "; the newest message is not JSON."))
        key = subjects and f"{topic}-key" in subjects and self._registry(f"/subjects/{topic}-key/versions/latest")
        more = [f"**Key:** {key.get('schemaType', 'AVRO')} `{_avro_type(json.loads(key['schema']))}`", ""] if key else []
        return card(topic, about, ["Field", "Type", "Description"], rows, more, defines=[topic])


def _newest(consumer, TopicPartition, topic: str, parts: list[int]):
    """The last message of the first partition that has one, or None."""
    for p in parts:
        low, high = consumer.get_watermark_offsets(TopicPartition(topic, p), timeout=10)
        if high > low:
            consumer.assign([TopicPartition(topic, p, high - 1)])
            msg = consumer.poll(10)
            consumer.unassign()
            if msg is not None and not msg.error():
                return msg
    return None


def _json(value: bytes | None):
    if not value or value[:1] == b"\0":  # empty, or Confluent-framed Avro/Protobuf
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None


def _json_type(v) -> str:
    return {bool: "boolean", int: "integer", float: "number", str: "string", dict: "object",
            list: "array", type(None): "null"}[type(v)]


def _retention(ms: str | None) -> str:
    if ms is None:
        return "default"
    if int(ms) < 0:
        return "forever"
    hours = int(ms) / 3_600_000
    return f"{hours / 24:g} days" if hours >= 24 else f"{hours:g} hours"


def _avro_type(t) -> str:
    if isinstance(t, list):
        kinds = [_avro_type(x) for x in t if x != "null"]
        return " | ".join(kinds) + (" (nullable)" if "null" in t else "")
    if isinstance(t, dict):
        if t.get("logicalType"):
            return f"{t['type']} ({t['logicalType']})"
        if t["type"] == "array":
            return f"array<{_avro_type(t['items'])}>"
        if t["type"] == "map":
            return f"map<{_avro_type(t['values'])}>"
        if t["type"] in ("record", "enum", "fixed"):
            return f"{t['type']} {t.get('name', '')}".strip()
        return _avro_type(t["type"])
    return str(t)


def _schema_fields(kind: str, schema: str) -> list[list[str]]:
    if kind == "AVRO":
        s = json.loads(schema)
        return [[f["name"], _avro_type(f["type"]), f.get("doc", "")] for f in s.get("fields", [])]
    if kind == "JSON":
        s = json.loads(schema)
        return [[k, str(v.get("type", "")), v.get("description", "")] for k, v in s.get("properties", {}).items()]
    return [["(schema)", kind, " ".join(schema.split())[:300]]]  # Protobuf: the definition, flattened
