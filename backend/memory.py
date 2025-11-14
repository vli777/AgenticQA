"""Conversation memory management with topic-level summaries and decay."""

import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from utils import get_embedding
from logger import logger


@dataclass
class MessageEntry:
    role: str
    content: str
    timestamp: float


@dataclass
class TopicMemory:
    id: str
    title: str
    messages: List[MessageEntry] = field(default_factory=list)
    summary: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    importance_score: float = 0.5
    embedding: Optional[np.ndarray] = None
    message_count: int = 0


@dataclass
class ConversationMemory:
    topics: Dict[str, TopicMemory] = field(default_factory=dict)
    active_topic_id: Optional[str] = None


class MemoryManager:
    def __init__(self):
        self._conversations: Dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()
        self.max_topics = 20
        self.similarity_threshold = 0.7
        self.half_life_seconds = 3600 * 24  # one day
        self.max_messages_per_topic = 12
        self.recent_message_keep = 4
        self.summary_llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct", temperature=0.0)

    def add_user_message(self, conversation_id: str, text: str, importance: float = 0.5) -> TopicMemory:
        with self._lock:
            conversation = self._conversations.setdefault(conversation_id, ConversationMemory())
            topic = self._match_or_create_topic(conversation, text, importance)
            topic.messages.append(MessageEntry(role="user", content=text, timestamp=time.time()))
            topic.last_active_at = time.time()
            topic.message_count += 1
            conversation.active_topic_id = topic.id
            self._compact_topic(topic)
            self._decay_topics(conversation)
            return topic

    def add_assistant_message(self, conversation_id: str, text: str):
        if not text:
            return
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation or not conversation.active_topic_id:
                return
            topic = conversation.topics.get(conversation.active_topic_id)
            if not topic:
                return
            topic.messages.append(MessageEntry(role="assistant", content=text, timestamp=time.time()))
            topic.last_active_at = time.time()
            topic.message_count += 1
            self._compact_topic(topic)

    def get_topic_context(self, conversation_id: str) -> List[Dict[str, str]]:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation or not conversation.active_topic_id:
                return []
            topic = conversation.topics.get(conversation.active_topic_id)
            if not topic:
                return []
            context: List[Dict[str, str]] = []
            if topic.summary:
                context.append({"role": "system", "content": f"Topic summary: {topic.summary}"})
            for entry in topic.messages[-self.recent_message_keep:]:
                context.append({"role": entry.role, "content": entry.content})
            return context

    def _match_or_create_topic(
        self,
        conversation: ConversationMemory,
        text: str,
        importance: float,
    ) -> TopicMemory:
        embedding = self._compute_embedding(text)
        best_topic: Optional[TopicMemory] = None
        best_score = 0.0

        if embedding is not None:
            for topic in conversation.topics.values():
                if topic.embedding is None:
                    continue
                score = float(np.dot(topic.embedding, embedding))
                if score > best_score:
                    best_score = score
                    best_topic = topic

        if best_topic and best_score >= self.similarity_threshold:
            topic = best_topic
            topic.importance_score = max(topic.importance_score, importance)
            if embedding is not None:
                topic.embedding = self._update_embedding(topic.embedding, embedding)
            return topic

        topic_id = uuid.uuid4().hex
        title = text[:80] or f"Topic {len(conversation.topics) + 1}"
        new_topic = TopicMemory(
            id=topic_id,
            title=title,
            importance_score=importance,
            embedding=embedding,
        )
        conversation.topics[topic_id] = new_topic
        conversation.active_topic_id = topic_id
        return new_topic

    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        try:
            vector = get_embedding(text)
            arr = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm == 0:
                return arr
            return arr / norm
        except Exception as exc:
            logger.warning(f"Failed to compute embedding for topic detection: {exc}")
            return None

    def _update_embedding(self, existing: np.ndarray, new_vector: np.ndarray) -> np.ndarray:
        combined = existing + new_vector
        norm = np.linalg.norm(combined)
        if norm == 0:
            return combined
        return combined / norm

    def _compact_topic(self, topic: TopicMemory):
        if len(topic.messages) <= self.max_messages_per_topic:
            return
        to_summarize = topic.messages[:-self.recent_message_keep]
        if not to_summarize:
            return
        try:
            summary_text = self._summarize_messages(topic.summary, to_summarize)
            topic.summary = summary_text
            topic.messages = topic.messages[-self.recent_message_keep:]
        except Exception as exc:
            logger.warning(f"Failed to summarize topic '{topic.title}': {exc}")

    def _summarize_messages(self, previous_summary: Optional[str], messages: List[MessageEntry]) -> str:
        history = "\n".join(f"{msg.role}: {msg.content}" for msg in messages)
        summary_prompt = (
            "Summarize the following conversation in a factual, concise way.\n"
            "Preserve user preferences, key facts, and unresolved questions.\n"
            f"Previous summary: {previous_summary or '(none)'}\n\n"
            f"Conversation:\n{history}\n\n"
            "Updated summary:"
        )
        response = self.summary_llm.invoke(summary_prompt)
        return response.content.strip()

    def _decay_topics(self, conversation: ConversationMemory):
        now = time.time()
        if len(conversation.topics) <= self.max_topics:
            return

        priorities = []
        for topic in conversation.topics.values():
            age = max(0.0, now - topic.last_active_at)
            decay = 0.5 ** (age / self.half_life_seconds)
            priority = topic.importance_score * decay
            priorities.append((priority, topic.id))

        priorities.sort()

        while len(conversation.topics) > self.max_topics and priorities:
            priority, topic_id = priorities.pop(0)
            if conversation.active_topic_id == topic_id:
                continue
            if priority < 0.05:
                del conversation.topics[topic_id]

    def get_context_text(self, conversation_id: str) -> str:
        history = self.get_topic_context(conversation_id)
        if not history:
            return ""
        parts = [f"{entry['role']}: {entry['content']}" for entry in history]
        return "\n".join(parts)


conversation_memory_manager = MemoryManager()
