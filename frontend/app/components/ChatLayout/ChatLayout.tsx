import React, { useState } from "react";
import { Box, Button, Modal, Typography, Link } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import Parameters from "../Parameters/Parameters"; // Adjust the import path as needed
import ParametersForm from "../Parameters/ParametersForm"; // Adjust the import path as needed
import { ChatContainer } from "../ChatContainer/ChatContainer"; // Adjust the import path as needed
import { ChatInput } from "../ChatInput/ChatInput"; // Adjust the import path as needed
import { ChatLayoutProps } from "../../types";
import { useStore } from "../../store"; // Adjust the import path as needed
import { create_ws_connection } from "../../utils";

export const ChatLayout = ({
  onSendMessage,
  messages,
  extend,
  onLogout,
  chatOnly = false,
  onFileAttach,
}: ChatLayoutProps) => {
  const [open, setOpen] = useState(false);

  const handleOpen = () => setOpen(true);
  const handleClose = () => setOpen(false);
  const {
    connectionStatus,
    isLeftPanelOpen,
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
    openaiKeyDebounced,
    mistralKeyDebounced,
    groqKeyDebounced,
    interaction_type,
    isInitialLoadComplete,
    rerank,
    setOpenaiKeyStatus,
    conversationID,
    setConversationID,
    username,
  } = useStore();

  const modalStyle = {
    position: "absolute",
    top: "20%",
    left: "20%",
    transform: "translate(-20%, -20%)",
    width: "80%",
    bgcolor: "background.paper",
    border: "2px solid #000",
    boxShadow: 24,
    p: 4,
    maxHeight: "90vh", // Set a maximum height
    overflowY: "auto", // Make it scrollable vertically
  };
  // console.log("myextend Layout", extend);

  type ConnectionStatus = "connected" | "disconnected" | "connecting"; // Example statuses
  type ButtonColor =
    | "inherit"
    | "primary"
    | "secondary"
    | "error"
    | "info"
    | "success"
    | "warning";

  const buttonColor: Record<ConnectionStatus, ButtonColor> = {
    connected: "success",
    connecting: "warning",
    disconnected: "error",
  };

  console.log('username', username);

  const buttonText: Record<ConnectionStatus, string> = {
    connected: username + " (Connected)",
    connecting: username + " (Connecting)",
    disconnected: username + " (Disconnected)",
  };
  return (
    <Box display="flex" flexDirection="column" height="80vh">
      <Box sx={{ flex: { xs: "0 1 auto", sm: "0 1 auto" } }}>
        <ParametersForm extend={extend} chatOnly={chatOnly} />
      </Box>

      {/* ChatContainer should dynamically adjust its height based on the available space */}
      <ChatContainer messages={messages} chatOnly={chatOnly} />

      {/* ChatInput might need adjustments to ensure it's not too invasive on small screens */}
      <Box sx={{ flex: "0 1 auto", minHeight: 0 }}>
        <ChatInput onSendMessage={onSendMessage} onFileAttach={onFileAttach} />
      </Box>

      <Box sx={{ position: "absolute", top: 16, right: 116 }}>
        <Button onClick={handleOpen}>Tutorial</Button>
      </Box>
      <Box sx={{ position: "absolute", top: 16, right: 16 }}>
        <Button onClick={onLogout}>Logout</Button>
      </Box>
      <Box
        sx={{ position: "absolute", top: 16, left: isLeftPanelOpen ? 250 : 50 }}
      >
        <Button
          color={buttonColor[connectionStatus]}
          onClick={() => {
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
          }}
        >
          {buttonText[connectionStatus]}
        </Button>
      </Box>
      <Modal
        open={open}
        onClose={handleClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box sx={modalStyle}>
          <Typography variant="h5" sx={{ color: "primary.main", mb: 1 }}>
            Tutoriel pour RAG: retrieval-augmented-generation
          </Typography>

          <Typography variant="body1" sx={{ color: "text.secondary" }}>
            <p>
              Un RAG est un LLM (Large language model) connecté à un moteur de
              recherche pour lui fournir des informations venant de notre base
              de donnée.{" "}
            </p>
            <p>
              <Link href="https://youtu.be/xbpLh9Ebvnc" target="_blank">
                Vidéo de présentation:
              </Link>{" "}
              <br />
              <Link href="https://youtu.be/q6JoNiMLOZ0" target="_blank">
                Vidéo tutorielle pour créer une clé API OpenAI:
              </Link>{" "}
            </p>


            <p>
              <b>Utilisation sans LLM:</b> choisir &quot;No Model&quot; dans la
              case LLM <br /> Vous utiliserez alors seulement le moteur de
              recherche <br />
            </p>
            <br />
            <p>
              <b>Utilisation avec LLM:</b> choisir un autre modèle dans la case
              LLM <br /> Vous utiliserez alors le moteur de recherche et le LLM
            </p>
            <br />
            <p>
              <b>Changer le nombre de documents envoyés au LLM: </b>
              <br />
              Changer la valeur dans la case &quot;N_docs_searched&quot; <br />
              Ceci correspond au nombre de documents envoyés au LLM pour lui
              permettre de répondre.
            </p>
            <br />
            <p>
              <b>Prix de l&apos;interaction: </b>
              <br />
              L&apos;utilisation du LLM est payante en fonction du nombre de
              mots envoyés. <br />
              Plus on envoie de documents, plus le tarif est élevé. <br />
              Le prix indiqué est une estimation maximale pour une interaction:
              pour une question posée
            </p>
            {/* <br />
            <p>
              <b>reranker: </b>
              <br />
              Le reranker est un modèle qui réordonne les résultats de la recherche<br />
              Cela permet d&apos;obtenir de meilleurs resultats mais est également plus couteux en temps <br />
            </p> */}
            <br />
            <p>
              <b>openai_key: </b>
              <br />
              Pour utiliser un LLM de openai, il faut renseigner une clé
              d&apos;accès dans la case &quot;openai_key&quot; <br />
              Vous pouvez créer une telle clé sur le site de{" "}
              <Link href="https://platform.openai.com/api-keys" target="_blank">
                OpenAI
              </Link>{" "}
              et ensuite{" "}
              <Link
                href="https://platform.openai.com/settings/organization/billing/overview"
                target="_blank"
              >
                ajouter de l&apos;argent
              </Link>{" "}
              sur le compte associé à la clé
              <br />
              Une telle clé est une chaine de caractère ressemblant à ceci:
              sk-LDKlkvmldkslKMDVlsdknvLIKJDSPIVspOMVLKIdvnsdvmLK <br />
              C&apos;est cette clé que vous devez renseigner dans la case
              &quot;openai_key&quot;
            </p>
            <br />
            <p>
              <b>Changer de librairie: </b>
              <br />
              Les librairies sont les banques de données auxquelles le LLM a accès. <br />
              Vous pouvez changer de librairie en ouvrant le menu déroulant à gauche. <br />
              Vous pouvez également créer une nouvelle librairie mais uniquement à partir de source_docs. <br />
              Pour cela, cliquez sur le bouton &quot;Create New Library&quot; <br />
            </p>
            <br />

            {extend && (
              <p>
                <b>mistral_key: (advanced users) </b>
                <br />
                Pour utiliser un LLM de mistral, il faut renseigner une clé
                d&apos;accès dans la case &quot;mistral_key&quot; <br />
                Vous pouvez créer une telle clé sur le site de{" "}
                <Link
                  href="https://console.mistral.ai/api-keys/"
                  target="_blank"
                >
                  Mistral
                </Link>
                <br />
                Une telle est clé une chaine de caractère ressemblant à ceci:
                ldkNVLDSKvdlkdjnvindDKVJNKJNDvnd
              </p>
            )}
            <br />

            {extend && (
              <p>
                <b>groq_key: (advanced users) </b>
                <br />
                Pour utiliser un LLM de mistral, il faut renseigner une clé
                d&apos;accès dans la case &quot;groq_key&quot; <br />
                Vous pouvez créer une telle clé sur le site de{" "}
                <Link href="https://console.groq.com/keys" target="_blank">
                  Groq
                </Link>
                <br />
                Une telle clé est une chaine de caractère ressemblant à ceci:
                gsk_KJNdKJSNlvkdslvcksDALKSDMelvmdsélvmdDVLméLMVdVöLKMdv
              </p>
            )}
          </Typography>
        </Box>
      </Modal>
    </Box>
  );
};
