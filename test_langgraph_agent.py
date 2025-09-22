#!/usr/bin/env python3
"""
Test script for the LangGraph agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph_agent.agent import graph, AgentState

def test_agent():
    """Test the LangGraph agent functionality."""
    print("🧪 Testing LangGraph Agent...")
    
    # Test 1: Import verification
    print("✅ Agent imported successfully")
    print(f"   Graph type: {type(graph)}")
    print(f"   State type: {type(AgentState)}")
    
    # Test 2: Create initial state
    initial_state = {
        "messages": [],
        "current_task": "idle",
        "prediction_results": {},
        "error": ""
    }
    
    print("✅ Initial state created")
    
    # Test 3: Run the graph
    try:
        print("🔄 Running agent workflow...")
        result = graph.invoke(initial_state)
        
        print("✅ Agent workflow completed successfully!")
        print(f"   Final task: {result.get('current_task', 'unknown')}")
        print(f"   Messages count: {len(result.get('messages', []))}")
        print(f"   Error: {result.get('error', 'none')}")
        
        # Print messages
        for i, msg in enumerate(result.get('messages', [])):
            print(f"   Message {i+1}: {msg.get('role', 'unknown')} - {msg.get('content', 'no content')[:50]}...")
            
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_agent()
    if success:
        print("\n🎉 All tests passed! LangGraph agent is ready to use.")
        print("\nTo start the development server, run:")
        print("   langgraph dev")
    else:
        print("\n❌ Tests failed. Please check the agent configuration.")
        sys.exit(1)
