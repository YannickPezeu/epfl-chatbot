
import { ChatMessage } from "../ChatMessage/ChatMessage";
import React, { useEffect, useRef, useState } from 'react';
import { Box } from '@mui/material';
import { ChatContainerProps } from '../../types'; // Adjust the import path as needed

export const ChatContainer: React.FC<ChatContainerProps> = ({ messages, chatOnly=false }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [lastMessageCount, setLastMessageCount] = useState(0);

  const scrollToBottom = () => {
    if (!containerRef.current) return;

    const { scrollTop, clientHeight, scrollHeight } = containerRef.current;
    const isNearBottom = scrollTop + clientHeight >= scrollHeight - 300;

    // Always scroll to bottom if a new message is added
    if (Object.keys(messages).length !== lastMessageCount) {
      containerRef.current.scrollTop = scrollHeight;
    } else if (isNearBottom) {
      // Scroll only if close to bottom and it's an update to an existing message
      containerRef.current.scrollTop = scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
    // Update the message count after scrolling
    setLastMessageCount(Object.keys(messages).length);
  }, [messages]); // Dependency array includes `messages`, so it triggers on message update

  return (
    <Box
      ref={containerRef} // Reference to the container for scroll checks
      id="chat-container"
      className="border rounded p-3"
      sx={{flex: { xs: '1 1 auto', sm: '0 1 auto' },  overflowY: "auto", height: chatOnly ? "70vh" : "60vh" }}
    >
      {Object.values(messages).map((msg) => (
        <ChatMessage
          key={msg.messageId}
          sender={msg.sender}
          message={msg.message}
          sources={msg.sources}
          messageType={msg.messageType}
          messageId={msg.messageId}
          nTokensInput={msg.nTokensInput}
          modelUsed={msg.modelUsed}
        />
      ))}
    </Box>
  );
};