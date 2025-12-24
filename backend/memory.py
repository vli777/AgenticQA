"""Conversation memory management with topic-level summaries and decay."""

import time
import uuid
import threading
import gzip
import json
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

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
    compressed_messages: Optional[str] = None  # Base64-encoded gzip compressed message history
    access_count: int = 0  # Number of times topic has been accessed (for LT memory consolidation)


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
        self.base_half_life_seconds = 3600 * 24  # 1 day (short-term memory)
        self.max_half_life_seconds = 3600 * 24 * 30  # 30 days (long-term memory cap)
        self.access_threshold_for_lt = 5  # Accesses needed to start LT consolidation
        self.max_messages_per_topic = 12
        self.recent_message_keep = 4
        self.summary_llm = ChatNVIDIA(model="meta/llama-4-maverick-17b-128e-instruct", temperature=0.0)

    def add_user_message(self, conversation_id: str, text: str, importance: float = 0.5) -> TopicMemory:
        with self._lock:
            conversation = self._conversations.setdefault(conversation_id, ConversationMemory())
            topic = self._match_or_create_topic(conversation, text, importance)

            # Decompress messages if topic was previously compressed
            self._restore_topic_if_compressed(topic)

            # Increment access count (for LT memory consolidation)
            topic.access_count += 1

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

            # Decompress messages if topic was previously compressed
            self._restore_topic_if_compressed(topic)

            # Increment access count (assistant messages also count as engagement)
            topic.access_count += 1

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

    def _get_topic_half_life(self, topic: TopicMemory) -> float:
        """
        Calculate dynamic half-life based on access patterns (mimics human memory consolidation).

        Short-term memory (infrequent access): Fast decay (~1 day)
        Long-term memory (frequent access): Slow decay (up to 30 days)

        Each access after threshold doubles the half-life (up to max).

        Returns:
            Half-life in seconds
        """
        # New or infrequent topics: use base half-life (short-term memory)
        if topic.access_count < self.access_threshold_for_lt:
            return self.base_half_life_seconds

        # Calculate LT memory half-life: start at 2x when threshold hit, then double every 5 more
        # access_count=5-9: 2 days, 10-14: 4 days, 15-19: 8 days, 20-24: 16 days, 25+: 30 days
        accesses_beyond_threshold = topic.access_count - self.access_threshold_for_lt
        multiplier = 2 ** (1 + accesses_beyond_threshold // self.access_threshold_for_lt)

        half_life = min(
            self.base_half_life_seconds * multiplier,
            self.max_half_life_seconds
        )

        return half_life

    def _compress_messages(self, messages: List[MessageEntry]) -> str:
        """
        Compress message history to save memory.

        Returns:
            Base64-encoded gzip compressed JSON string
        """
        try:
            # Serialize messages to JSON
            messages_data = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in messages
            ]
            json_str = json.dumps(messages_data)

            # Compress with gzip
            compressed = gzip.compress(json_str.encode('utf-8'))

            # Encode to base64 for safe storage
            encoded = base64.b64encode(compressed).decode('ascii')

            logger.debug(f"Compressed {len(messages)} messages: {len(json_str)} -> {len(compressed)} bytes")
            return encoded
        except Exception as exc:
            logger.error(f"Failed to compress messages: {exc}")
            return None

    def _decompress_messages(self, compressed_data: str) -> List[MessageEntry]:
        """
        Decompress message history.

        Args:
            compressed_data: Base64-encoded gzip compressed JSON string

        Returns:
            List of MessageEntry objects
        """
        try:
            # Decode from base64
            compressed = base64.b64decode(compressed_data.encode('ascii'))

            # Decompress with gzip
            json_str = gzip.decompress(compressed).decode('utf-8')

            # Deserialize from JSON
            messages_data = json.loads(json_str)

            messages = [
                MessageEntry(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg["timestamp"]
                )
                for msg in messages_data
            ]

            logger.debug(f"Decompressed {len(messages)} messages from {len(compressed)} bytes")
            return messages
        except Exception as exc:
            logger.error(f"Failed to decompress messages: {exc}")
            return []

    def _restore_topic_if_compressed(self, topic: TopicMemory):
        """
        Restore full message history if topic has compressed messages.
        This is called when a previously inactive topic becomes active again.
        """
        if not topic.compressed_messages:
            return

        try:
            # Decompress old messages
            old_messages = self._decompress_messages(topic.compressed_messages)

            if old_messages:
                # Combine old compressed messages with current active messages
                # Sort by timestamp to maintain chronological order
                all_messages = old_messages + topic.messages
                all_messages.sort(key=lambda m: m.timestamp)

                # Restore to active memory
                topic.messages = all_messages
                topic.compressed_messages = None  # Clear compressed storage

                logger.info(f"Restored {len(old_messages)} compressed messages for topic '{topic.title}'")
        except Exception as exc:
            logger.warning(f"Failed to restore compressed messages for topic '{topic.title}': {exc}")

    def _compact_topic(self, topic: TopicMemory):
        if len(topic.messages) <= self.max_messages_per_topic:
            return
        to_summarize = topic.messages[:-self.recent_message_keep]
        if not to_summarize:
            return
        try:
            # Create summary of old messages
            summary_text = self._summarize_messages(topic.summary, to_summarize)
            topic.summary = summary_text

            # Compress old messages instead of deleting them
            # If we already have compressed messages, decompress them first
            existing_compressed = []
            if topic.compressed_messages:
                existing_compressed = self._decompress_messages(topic.compressed_messages)

            # Combine existing compressed messages with the ones we're about to compress
            all_old_messages = existing_compressed + to_summarize

            # Compress the full history
            compressed_data = self._compress_messages(all_old_messages)
            if compressed_data:
                topic.compressed_messages = compressed_data
                logger.info(f"Compressed {len(all_old_messages)} messages for topic '{topic.title}'")

            # Keep only recent messages in active memory
            topic.messages = topic.messages[-self.recent_message_keep:]
        except Exception as exc:
            logger.warning(f"Failed to compact topic '{topic.title}': {exc}")

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

            # Dynamic half-life based on access patterns (ST vs LT memory)
            half_life = self._get_topic_half_life(topic)

            decay = 0.5 ** (age / half_life)
            priority = topic.importance_score * decay
            priorities.append((priority, topic.id, half_life))

        priorities.sort()

        # Log decay information for transparency
        for priority, topic_id, half_life in priorities[:3]:  # Show bottom 3
            topic = conversation.topics[topic_id]
            age_days = (now - topic.last_active_at) / 86400
            logger.debug(
                f"Topic '{topic.title[:40]}': priority={priority:.3f}, "
                f"age={age_days:.1f}d, accesses={topic.access_count}, "
                f"half_life={half_life/86400:.1f}d"
            )

        while len(conversation.topics) > self.max_topics and priorities:
            priority, topic_id, half_life = priorities.pop(0)
            if conversation.active_topic_id == topic_id:
                continue
            if priority < 0.05:
                topic = conversation.topics[topic_id]
                logger.info(
                    f"Removing topic '{topic.title[:40]}' (priority={priority:.3f}, "
                    f"accesses={topic.access_count})"
                )
                del conversation.topics[topic_id]

    def get_context_text(self, conversation_id: str) -> str:
        history = self.get_topic_context(conversation_id)
        if not history:
            return ""
        parts = [f"{entry['role']}: {entry['content']}" for entry in history]
        return "\n".join(parts)

    def get_full_topic_history(self, conversation_id: str, include_compressed: bool = True) -> List[Dict[str, str]]:
        """
        Get the complete conversation history for the active topic, including compressed messages.

        Args:
            conversation_id: Conversation identifier
            include_compressed: If True, decompress and include old messages

        Returns:
            List of message dictionaries in chronological order
        """
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation or not conversation.active_topic_id:
                return []

            topic = conversation.topics.get(conversation.active_topic_id)
            if not topic:
                return []

            all_messages = []

            # Include compressed messages if requested
            if include_compressed and topic.compressed_messages:
                try:
                    old_messages = self._decompress_messages(topic.compressed_messages)
                    all_messages.extend(old_messages)
                except Exception as exc:
                    logger.warning(f"Failed to decompress messages for full history: {exc}")

            # Add current active messages
            all_messages.extend(topic.messages)

            # Sort by timestamp to ensure chronological order
            all_messages.sort(key=lambda m: m.timestamp)

            # Convert to dictionary format
            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in all_messages
            ]

    def get_topic_stats(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get statistics about the current topic's memory usage.

        Returns:
            Dictionary with stats about active messages, compressed messages, etc.
        """
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if not conversation or not conversation.active_topic_id:
                return {}

            topic = conversation.topics.get(conversation.active_topic_id)
            if not topic:
                return {}

            compressed_count = 0
            if topic.compressed_messages:
                try:
                    old_messages = self._decompress_messages(topic.compressed_messages)
                    compressed_count = len(old_messages)
                except Exception:
                    pass

            # Calculate current half-life and memory type
            half_life = self._get_topic_half_life(topic)
            memory_type = "long-term" if topic.access_count >= self.access_threshold_for_lt else "short-term"

            return {
                "topic_id": topic.id,
                "topic_title": topic.title,
                "active_messages": len(topic.messages),
                "compressed_messages": compressed_count,
                "total_messages": len(topic.messages) + compressed_count,
                "has_summary": bool(topic.summary),
                "importance_score": topic.importance_score,
                "access_count": topic.access_count,
                "memory_type": memory_type,
                "half_life_days": half_life / 86400,
                "age_seconds": time.time() - topic.created_at,
                "inactive_seconds": time.time() - topic.last_active_at
            }


conversation_memory_manager = MemoryManager()
