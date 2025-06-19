import React from "react";
import styles from "./ChatMessage.module.css"; // Ajustez le chemin selon votre structure
import { Grid, Typography, Link, Paper, Box } from "@mui/material";

import { ChatMessageProps, Source } from "../../types"; // Adjust the import path as needed
import { useStore } from "../../store";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faExternalLinkAlt } from "@fortawesome/free-solid-svg-icons";
import { Collapse, IconButton, Tooltip } from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { RotateCcw, ChevronDown } from "lucide-react";
import { create_ws_connection } from "../../utils";
import { v4 as uuidv4 } from "uuid";



const transform_source_format = (text: string) => {
  // Regular expression to match numbers within square brackets, globally.
  const regex = /\[\s*\d+(?:\s*,\s*\d+)*\s*\]/g;

  const newText = text.replace(regex, (match) => {
    // Extract the matched string, remove square brackets, split by commas, and trim whitespace
    const numbersString = match.slice(1, -1);
    const numbersArray = numbersString.split(",").map((str) => str.trim());

    // Transform each number into the format "[number]" and combine them into a single string
    // Here, each match is transformed individually, and the string to replace it is returned directly
    const transformed = numbersArray.map((number) => `[${number}]`).join(", ");
    return transformed;
  });
  return newText;
};

// Fonction pour transformer les références en liens cliquables
const renderMessageWithLinks = (
  text: string,
  sources: Source[],
  onClickSource: (pdfId: number, pdfPageNumber: number) => void,
  BASE_URL: string
) => {
  // Process text modifications once
  const processedText = transform_source_format(text);
  
  // Create final elements array
  const elements: JSX.Element[] = [];
  let currentText = processedText;
  let key = 0;

  // Process bold sections
  while (currentText.length > 0) {
    const boldStart = currentText.indexOf('**');
    if (boldStart === -1) {
      // No more bold sections, process remaining text for sources
      elements.push(...processSourceLinks(currentText, sources, onClickSource, key));
      break;
    }

    // Add text before bold section
    if (boldStart > 0) {
      elements.push(...processSourceLinks(currentText.slice(0, boldStart), sources, onClickSource, key));
      key += 1;
    }

    // Find end of bold section
    const boldEnd = currentText.indexOf('**', boldStart + 2);
    if (boldEnd === -1) {
      // Unclosed bold marker, treat rest as normal text
      elements.push(...processSourceLinks(currentText.slice(boldStart), sources, onClickSource, key));
      break;
    }

    // Add bold section
    elements.push(
      <strong key={`bold-${key}`}>
        {currentText.slice(boldStart + 2, boldEnd)}
      </strong>
    );
    key += 1;

    // Continue with remaining text
    currentText = currentText.slice(boldEnd + 2);
  }

  return <>{elements}</>;
};

const processSourceLinks = (
  text: string,
  sources: Source[],
  onClickSource: (pdfId: number, pdfPageNumber: number) => void,
  baseKey: number
): JSX.Element[] => {
  return text
    .split(/(\[\d+\])/)
    .filter(Boolean)
    .map((part, index) => {
      const match = part.match(/^\[(\d+)\]$/);
      if (!match) return <span key={`text-${baseKey}-${index}`}>{part}</span>;

      const sourceIndex = parseInt(match[1], 10);
      if (sourceIndex >= 0 && sourceIndex < sources.length) {
        const source = sources[sourceIndex];
        return (
          <span
            key={`source-${baseKey}-${index}`}
            onClick={() => onClickSource(
              source.source_doc_id,
              parseInt(source.page_number) + 1
            )}
            style={{
              cursor: "pointer",
              textDecoration: "underline",
              fontWeight: "bold",
            }}
          >
            [{sourceIndex}]
          </span>
        );
      }
      
      return <span key={`invalid-${baseKey}-${index}`}>{part}</span>;
    });
};
export const ChatMessage = ({
  sender,
  message,
  sources,
  messageType,
  nTokensInput,
  messageId,
  modelUsed,
}: ChatMessageProps) => {
  const [openSourcesMap, setOpenSourcesMap] = React.useState<any>({});

  const handleToggleSources = (messageId: string) => {
    setOpenSourcesMap((prev: any) => ({
      ...prev,
      [messageId]: !prev[messageId],
    }));
  };

  const {
    pdfId,
    setPdfId,
    setPdfPageNumber,
    setIsDrawerOpen,
    BASE_URL,
    price_token,
    selectedModel,
    selectedLibrary,
    socket,
    isInitialLoadComplete,
    nDocumentsSearchedNoLLMDebounced,
    nDocumentsSearchedDebounced,
    setMessages,
    setSocket,
    base64Credentials,
    setBASE_URL,
    setBASE_URL_WS,
    selectedEmbeddingModel,
    setConnectionStatus,
    BASE_URL_WS,
    openaiKeyDebounced,
    mistralKeyDebounced,
    groqKeyDebounced,
    interaction_type,
    rerank,
    messages,
    setOpenaiKeyStatus,
    conversationID,
    setConversationID
  } = useStore();

  const reloadMessage = () => {

    // get the index of the message in the array
    const messageIndex = messages.findIndex((msg) => msg.messageId === messageId);

    // get last message with message_type == user_input and get the index in the array
    const lastUserInputMessage = messages.slice(0, messageIndex).reverse().find((msg) => msg.messageType === "user_input");
    const lastUserInputMessageIndex = messages.findIndex((msg) => msg.messageId === lastUserInputMessage?.messageId);


    if (!lastUserInputMessage) {
      console.error("No user input message found");
      alert("No user input message found");
      return;
    }
    const message_content = lastUserInputMessage.message.message_content;
  

    if (selectedLibrary === null) {
      console.error("No library selected");
      alert("Please select a library to continue");
      return;
    }
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      console.log("Socket not open, trying to reconnect");
      if (isInitialLoadComplete) {
        create_ws_connection(
          selectedLibrary,
          selectedModel,
          nDocumentsSearchedNoLLMDebounced,
          nDocumentsSearchedDebounced,
          setMessages,
          setSocket,
          base64Credentials,
          setBASE_URL,
          setBASE_URL_WS,
          selectedEmbeddingModel,
          setConnectionStatus,
          BASE_URL,
          BASE_URL_WS,
          setConversationID,
          openaiKeyDebounced,
          mistralKeyDebounced,
          groqKeyDebounced,
          interaction_type,
          rerank,
          setOpenaiKeyStatus,
          conversationID,
        );
      }
      // wait for connection to be established

      console.error("Failed to send message: please try again");
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          sender: "Bot",
          message: {
            message_content: "Failed to send message: please try again",
          },
          messageId: uuidv4(),
          sources: [],
          messageType: "error",
          modelUsed: selectedModel,
        },
      ]);
    } else {
      socket.send(
        JSON.stringify({
          user_input: message_content,
          interaction_type: interaction_type,
          reload_message: true,
        })
      );
      setMessages((prevMessages) => {
        prevMessages = prevMessages.slice(0, lastUserInputMessageIndex);

        prevMessages =[
        ...prevMessages,
        {
          sender: "You",
          message: { message_content: message_content },
          messageId: uuidv4(),
          sources: [],
          messageType: "user_input",
          modelUsed: selectedModel,
        },
      ]
      return prevMessages;
    });
    }
  };

  const onClickSource = (pdfId: number, pdfPageNumber: number) => {
    setPdfId(pdfId);
    setPdfPageNumber(pdfPageNumber);
    setIsDrawerOpen(true);
  };
  if (sources === undefined) {
    sources = [];
  }

  if (messageType === "error") {
    return (
      <div className={`${styles.errorMessageContainer}`}>
        <div className={`${styles.errorMessage}`}>
          {message.message_content}
        </div>
      </div>
    );
  }

  const openInNewTab = (url: string) => {
    const newWindow = window.open(url, "_blank", "noopener,noreferrer");
    if (newWindow) newWindow.opener = null;
  };

  if (messageType === "No_Model") {
    return (
      <div className={`${sender === "You" ? styles.userMessageContainer : styles.botMessageContainer}`}>
        <div className={`${sender === "You" ? styles.userMessage : styles.botMessage}`}>
          {sources?.map((source, index) => (
            <div key={index} style={{ display: 'flex', alignItems: 'center', marginBottom: '8px', width: '100%', flexWrap: 'nowrap' }}>
              <Tooltip title="Opens saved PDF at the right page">
                <span
                  onClick={() => onClickSource(source.source_doc_id, parseInt(source.page_number) + 1)}
                  style={{ 
                    cursor: 'pointer',
                    flexGrow: 1,
                    whiteSpace: 'normal'
                  }}
                  className="hover:font-bold"
                >
                  {source.title.replace(/_/g, " ")} (page {parseInt(source.page_number) + 1}){": "}
                </span>
              </Tooltip>
  
              {source.url && source.url !== 'unknown' && (
                <Tooltip title="Opens up-to-date URL">
                  <span style={{ flexShrink: 0, marginLeft: '8px', display: 'inline-flex', alignItems: 'center' }}>
                    <FontAwesomeIcon
                      icon={faExternalLinkAlt}
                      onClick={(e) => {
                        e.stopPropagation();
                        openInNewTab(source.url);
                      }}
                      className="cursor-pointer hover:scale-110 transition-transform"
                    />
                  </span>
                </Tooltip>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  }
  else if (messageType === "tool_input") {
    return (
      <div
        className={
          sender === "You"
            ? styles.userMessageContainer
            : styles.botMessageContainer
        }
      >
        <div
          className={sender === "You" ? styles.userMessage : styles.botMessage}
        >
          <strong>Tool name:</strong> {message.tool_name}
          <br />
          <strong>Tool input:</strong> {message.tool_input}
        </div>
      </div>
    );
  } else {
    
    return (
      <div
        className={`${
          sender === "You"
            ? styles.userMessageContainer
            : styles.botMessageContainer
        }`}
      >
        <div
          className={`${
            sender === "You" ? styles.userMessage : styles.botMessage
          }`}
        >
          {message.message_content !== undefined &&
            message.message_content
              .split("\n")
              .map((line, index) => (
                <div key={index}>
                  {renderMessageWithLinks(
                    line,
                    sources,
                    onClickSource,
                    BASE_URL
                  )}
                </div>
              ))}

          {messageType === "response_without_tool" && sources.length > 0 && (
            <>
              <Box display="flex" alignItems="center" mt={2}>
                Sources:
                <IconButton
                  onClick={() => handleToggleSources(messageId)}
                  aria-expanded={openSourcesMap[messageId] || false}
                  aria-label="show more"
                  size="small"
                  sx={{
                    transform: openSourcesMap[messageId]
                      ? "rotate(180deg)"
                      : "rotate(0deg)",
                    transition: "transform 0.3s",
                  }}
                >
                  <ChevronDown size={20} />
                </IconButton>
              </Box>

              <Collapse
                in={openSourcesMap[messageId] || false}
                timeout="auto"
                unmountOnExit
              >
                {sources.map((source, index) => (
                  <div
                    key={index}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      textDecoration: "underline",
                      width: "100%",
                    }}
                  >
                    <Tooltip title="Opens saved PDF at the right page">
                      <div
                        onClick={() =>
                          onClickSource(
                            source.source_doc_id,
                            parseInt(source.page_number) + 1
                          )
                        }
                        style={{ cursor: "pointer" }}
                      >
                        <strong>[{index}]</strong>:{" "}
                        {source.title.replace(/_/g, " ")} (page{" "}
                        {parseInt(source.page_number) + 1})
                      </div>
                    </Tooltip>
                    {source.url && source.url != 'unknown' &&
                    <Tooltip title="Opens up to date url">
                     <FontAwesomeIcon
                      icon={faExternalLinkAlt}
                      onClick={(e) => {
                        e.stopPropagation();
                        openInNewTab(source.url);
                      }}
                      style={{ cursor: "pointer" }}
                    />
                    </Tooltip>
                    }
                  </div>
                ))}
              </Collapse>
            </>
          )}

          <Box
            display="flex"
            justifyContent="space-between"
            alignItems="flex-end"
            mt={2}
          >
            {messageType == "response_without_tool" && (
              <Tooltip title="Reload Message">
                <IconButton
                  onClick={() => reloadMessage()}
                  color="default"
                  aria-label="reload message"
                  size="small"
                >
                  <RotateCcw size={16} />
                </IconButton>
              </Tooltip>
            )}

            {nTokensInput && (
              <Typography variant="body2">
                Cost:{" "}
                {Number(nTokensInput * price_token[modelUsed]).toPrecision(2)}{" "}
                CHF
              </Typography>
            )}
          </Box>
        </div>
      </div>
    );
  }
};
