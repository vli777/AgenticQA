"""Quick smoke test for memory module changes."""

import sys
sys.path.insert(0, 'E:\\Git\\AgenticQA\\backend')

from memory import conversation_memory_manager

def test_memory_smoke():
    """Test basic memory operations with new dynamic decay."""
    conv_id = "test_user_123"

    print("Test 1: Add messages and check access_count...")
    # Add first message (should create new topic)
    topic = conversation_memory_manager.add_user_message(
        conv_id,
        "What is Python?",
        importance=0.5
    )
    print(f"[OK] Created topic: {topic.title[:40]}")
    print(f"  Access count: {topic.access_count}")
    assert topic.access_count >= 1, "Access count should be >= 1"

    # Add assistant response
    conversation_memory_manager.add_assistant_message(
        conv_id,
        "Python is a programming language."
    )
    print(f"[OK] Added assistant message")
    print(f"  Access count after response: {topic.access_count}")

    # Add more messages to trigger LT memory
    print("\nTest 2: Trigger LT memory consolidation...")
    for i in range(4):
        conversation_memory_manager.add_user_message(
            conv_id,
            f"Tell me more about Python feature {i}",
            importance=0.5
        )
        conversation_memory_manager.add_assistant_message(
            conv_id,
            f"Feature {i} explanation."
        )

    print(f"  Access count after multiple messages: {topic.access_count}")

    # Test 3: Get stats with new fields
    print("\nTest 3: Check stats with new fields...")
    stats = conversation_memory_manager.get_topic_stats(conv_id)
    print(f"[OK] Got stats:")
    print(f"  Topic: {stats['topic_title'][:40]}")
    print(f"  Access count: {stats['access_count']}")
    print(f"  Memory type: {stats['memory_type']}")
    print(f"  Half-life (days): {stats['half_life_days']:.1f}")

    assert 'access_count' in stats, "access_count missing from stats"
    assert 'memory_type' in stats, "memory_type missing from stats"
    assert 'half_life_days' in stats, "half_life_days missing from stats"

    # Test 4: Verify memory type transition
    print("\nTest 4: Verify ST -> LT transition...")
    if stats['access_count'] < 5:
        assert stats['memory_type'] == 'short-term', f"Should be short-term, got {stats['memory_type']}"
        assert stats['half_life_days'] == 1.0, f"Should be 1 day, got {stats['half_life_days']}"
        print("  [OK] Correctly in short-term memory (< 5 accesses)")
    else:
        assert stats['memory_type'] == 'long-term', f"Should be long-term, got {stats['memory_type']}"
        assert stats['half_life_days'] > 1.0, f"Should be > 1 day, got {stats['half_life_days']}"
        print(f"  [OK] Correctly in long-term memory (>= 5 accesses)")

    print("\n[PASS] All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_memory_smoke()
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
