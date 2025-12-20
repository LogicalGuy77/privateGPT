# Private-GPT UI Testing Guide

## Running the Application

### Mock Mode (No Model Loading)
```bash
cd /home/harshit/coding/private-gpt/private-gpt-app
PYTHONPATH=src uv run python -m private_gpt_app.main --mock --dev
```

### Testing Checklist

✅ **Application Startup**
- Window opens with "Private-GPT" title
- Sidebar visible on the left (250-400px wide)
- Chat area visible on the right
- Input area at the bottom

✅ **UI Components**
- Sidebar shows "Chat History" header
- "+ New Chat" button present
- Session list shows placeholder sessions
- Status label at bottom of sidebar

✅ **Chat Area**
- Welcome message displayed initially
- Input field with placeholder text
- "🎭 Mock Mode" indicator visible
- "Clear" and "Send" buttons present

✅ **Interaction Tests**
1. **Type and Send:**
   - Type a message in the input field
   - Click "Send" or press Ctrl+Enter
   - User message appears as blue bubble on the right
   - Mock response appears as dark gray bubble on the left (with streaming effect)

2. **Token Streaming:**
   - Watch the mock response appear character-by-character
   - Markdown formatting should render after streaming completes

3. **Clear Button:**
   - Type text in input
   - Click "Clear"
   - Input should empty

4. **New Chat:**
   - Send a few messages
   - Click "+ New Chat"
   - Chat area clears and shows welcome message again

5. **Hot Reload (Dev Mode):**
   - Edit `src/private_gpt_app/ui/styles.qss`
   - Save the file
   - Stylesheet should reload automatically
   - Console should show "🔄 Reloaded stylesheet"

✅ **Visual Verification**
- Background: Deep black (#0A0A0A)
- Sidebar: Charcoal grey (#1A1A1A)
- User bubbles: Blue (#2563EB)
- Assistant bubbles: Dark grey (#1A1A1A) with border
- Text: High contrast white/off-white
- Smooth scrolling in chat area

## Known Issues to Fix Later
- [ ] Message bubble height auto-adjustment needs refinement
- [ ] Keyboard shortcuts (Ctrl+N for new chat) not yet implemented
- [ ] Session persistence not yet implemented
- [ ] Markdown code blocks need syntax highlighting

## Next Steps (Phase 1 Continued)
- [ ] Integrate llama.cpp with lazy loading
- [ ] Add GPU detection and VRAM validation
- [ ] Implement actual token streaming from LLM
- [ ] Add crash recovery auto-save
