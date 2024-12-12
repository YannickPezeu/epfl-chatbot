import React from "react";
import { Grid, IconButton } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";

import { v4 as uuidv4 } from "uuid";
import PDFDrawer from "../Drawer/Drawer";
import { Source } from "../../types"; // Adjust the import path as needed // Ajustez le chemin d'import selon votre structure de dossier
import { create_ws_connection } from "../../utils";
import { useStore } from "../../store"; // Adjust the import path as needed
import { ChatLayout } from "../ChatLayout/ChatLayout";
import { useEffect, useState } from "react";
import LeftPanel from "../LeftPannel/LeftPannel"; // Import the new LeftPanel component
import CreateLibraryModal from "../CreateLibraryModal/CreateLibraryModal";
import ErrorModal from "../ErrorModal/ErrorModal"; // Adjust the import path as needed

const Main: React.FC<{ extend: boolean; chatOnly?: boolean }> = ({
  extend,
  chatOnly = false,
}) => {
  const {
    messages,
    setMessages,
    socket,
    setSocket,
    selectedModel,
    nDocumentsSearchedNoLLMDebounced,
    nDocumentsSearchedDebounced,
    isDrawerOpen,
    setIsDrawerOpen,
    base64Credentials,
    setBASE_URL,
    BASE_URL,
    BASE_URL_WS,
    setBASE_URL_WS,
    selectedEmbeddingModel,
    openaiKeyDebounced,
    mistralKeyDebounced,
    groqKeyDebounced,
    setConnectionStatus,
    isInitialLoadComplete,
    selectedLibrary: selectedLibrary,
    setSelectedLibrary: setSelectedLibrary,
    isLoggedIn,
    setIsLoggedIn,
    rerank,
    isLeftPanelOpen,
    setIsLeftPanelOpen,
    interaction_type,
    filesAttached,
    setFilesAttached,
    libraryCreationProgress,
    setLibraryCreationProgress,
    setOpenaiKeyStatus,
    conversationID,
    setConversationID,
    libraries,
    setLibraries,
    username,
    setUsername,
  } = useStore();

  const fetchUsername = async () => {
    try {
      const response = await fetch(`${BASE_URL}/auth/get-username`, {
        method: "GET",
        credentials: "include",
      });

      if (response.ok) {
        const data = await response.json();
        setUsername(data.username);
      } else {
        console.error("Failed to fetch username");
      }
    } catch (error) {
      console.error("Error fetching username:", error);
    }
  };

  useEffect(() => {
    fetchUsername();
  }, []);

  if (chatOnly && selectedLibrary !== "no_library") {
    setSelectedLibrary("no_library");
  }

  // const [isLeftPanelOpen, setIsLeftPanelOpen] = useState(false);
  // const [libraries, setLibraries] = useState(["LEX", "RH"]);
  const [isCreateLibraryModalOpen, setIsCreateLibraryModalOpen] =
    useState(false);
  const [error, setError] = useState("");
  const [isErrorModalOpen, setIsErrorModalOpen] = useState<boolean>(false);
  const setErrorAndOpenModal = (errorMessage: string) => {
    setError(errorMessage);
    setIsErrorModalOpen(true);
  };

  //update libraries via server call
  const fetchLibraries = async () => {
    try {
      const response = await fetch(`${BASE_URL}/libraries/check_user_libraries`, {
        method: "GET",
        credentials: "include",
      });

      if (response.ok) {
        const data = await response.json();
        setLibraries(data.libraries);
      } else {
        console.error("Failed to fetch libraries");
      }
    } catch (error) {
      console.error("Error fetching libraries:", error);
    }
  };

  useEffect(() => {
    fetchLibraries();
  }, []);

  const handleLibrarySelect = (library: string) => {
    setSelectedLibrary(library);
    console.log(`Selected library: ${library}`);
  };

  const handleCreateNewLibrary = async (
    library_name: string,
    files: File[],
    selectedEmbeddingModel: string,
    specialPrompt: string
  ) => {
    return new Promise<void>(async (resolve, reject) => {
      try {
        const formData = new FormData();
        console.log('Files to add:', files);
        
        files.forEach((file, index) => {
          console.log(`Adding file ${index}:`, file);
          formData.append("files", file);
        });
        
        console.log('Adding library_name:', library_name);
        formData.append("library_name", library_name);
        
        console.log('Adding model_name:', selectedEmbeddingModel);
        formData.append("model_name", selectedEmbeddingModel);
        
        console.log('Adding special_prompt:', specialPrompt);
        formData.append("special_prompt", specialPrompt);

        // Log formData entries
        console.log('FormData entries:');
        for (let pair of formData.entries()) {
            console.log('entry ',pair[0], pair[1]);
        }
        console.log('formData', formData);

        const urlWithParameter = `${BASE_URL}/libraries/create`;
        
        const response = await fetch(urlWithParameter, {
          method: "POST",
          body: formData,
          credentials: "include",
          headers: {
            "X-Requested-With": "XMLHttpRequest",
          },
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText);
        }

        const data = await response.json();
        
        if (data.status !== "Started") {
          throw new Error("Library creation didn't start properly");
        }

        console.log("Library creation started");
        const taskId = data.task_id;
        const eventSource = new EventSource(`${BASE_URL}/progress/${taskId}`);

        eventSource.onerror = (event) => {
          console.error("EventSource error:", event);
          eventSource.close();
          reject(new Error("EventSource error"));
        };

        eventSource.onmessage = (event) => {
          console.log("message:", event.data);
          const progressData = JSON.parse(event.data);
          if (progressData.status === "Error") {
            console.error("Error during library creation:", progressData.error);
            eventSource.close();
            reject(new Error(progressData.error));
            return;
          }

          const newLibraryCreationProgress: Array<{
            [library_name: string]: {
              'progress': number,
              'price': number,
            };
          }> = [...libraryCreationProgress];

          const existingIndex = newLibraryCreationProgress.findIndex(
            (item) => Object.keys(item)[0] === library_name
          );

          if (existingIndex !== -1) {
            newLibraryCreationProgress[existingIndex] = { [library_name]: {
              'progress': progressData.progress},
              'price': progressData.price,
           };
          } else {
            newLibraryCreationProgress.push({ [library_name]: {
              'progress': progressData.progress,
              'price': progressData.price,
            } });
          }

          setLibraryCreationProgress(newLibraryCreationProgress);

          if (progressData.progress === 100) {
            eventSource.close();
            fetchLibraries();
            resolve();
          }
        };

      } catch (error) {
        console.error("Error during library creation:", error);
        reject(error);
      }
    });
  };

  const handleDeleteLibrary = async (libraryName: string) => {
    try {
      console.log("delete library:", libraryName);
      const response = await fetch(`${BASE_URL}/libraries/${libraryName}`, {
        method: "DELETE",  // Changed from POST to DELETE
        headers: {
            "X-Requested-With": "XMLHttpRequest",
        },
        // No body needed since library name is in the URL now
        credentials: "include",
    });
  
      if (response.ok) {
        const data = await response.json();
        console.log(data.message); // Log the success message from the server
        // Refresh the list of libraries
        fetchLibraries();
        // You might want to update some state to reflect the deletion
        // For example, if you're keeping track of selected libraries:
        // setSelectedLibrary(null);
      } else {
        const errorText = await response.text();
        console.error("Error response:", errorText);
        let errorMessage: string;
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.detail || `Failed to delete library "${libraryName}". Please try again.`;
        } catch (e) {
          errorMessage = `Failed to delete library "${libraryName}". Please try again.`;
        }
        throw new Error(errorMessage);
      }
    } catch (error: unknown) {
      console.error("Error during library deletion:", error);
      let errorMessage: string;
      if (error instanceof Error) {
        errorMessage = error.message;
      } else if (typeof error === 'object' && error !== null && 'message' in error) {
        errorMessage = (error as { message: string }).message;
      } else {
        errorMessage = `An unexpected error occurred while deleting library "${libraryName}"`;
      }
      setErrorAndOpenModal(errorMessage);
    }
  };

  const handleLogout = async () => {
    try {
      const response = await fetch(`${BASE_URL}/auth/logout`, {
        method: "POST",
        credentials: "include",
      });

      if (response.ok) {
        document.cookie =
          "session_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
        setIsLoggedIn(false);
      } else {
        console.error("Failed to log out");
      }
    } catch (error) {
      console.error("Error logging out:", error);
    }
  };

  const toggleDrawer = (open: boolean) => () => {
    setIsDrawerOpen(open);
  };

  useEffect(() => {
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

    return () => {
      if (socket) {
        socket.close();
        setConnectionStatus("disconnected");
      }
    };
  }, [
    selectedModel,
    nDocumentsSearchedNoLLMDebounced,
    nDocumentsSearchedDebounced,
    openaiKeyDebounced,
    mistralKeyDebounced,
    selectedEmbeddingModel,
    groqKeyDebounced,
    selectedLibrary,
    rerank,
  ]);

  const handleSendMessage = (message: string) => {
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
          user_input: message,
          interaction_type: interaction_type,
          reload_message: false,
        })
      );
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          sender: "You",
          message: { message_content: message },
          messageId: uuidv4(),
          sources: [],
          messageType: "user_input",
          modelUsed: selectedModel,
        },
      ]);
      setFilesAttached([]);
    }
  };

  const handleFileAttach = async (files: File[]) => {
    console.log("Files attached:", files);
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      console.error("WebSocket is not open");
      return;
    }
  
    const filePromises = files.map(async (file) => {
      try {
        const arrayBuffer = await file.arrayBuffer();
  
        const header = JSON.stringify({
          type: "file",
          filename: file.name,
          size: arrayBuffer.byteLength,
        });
  
        const headerBuffer = new TextEncoder().encode(header);
        const combinedBuffer = new ArrayBuffer(
          4 + headerBuffer.byteLength + arrayBuffer.byteLength
        );
        const view = new DataView(combinedBuffer);
  
        view.setUint32(0, headerBuffer.byteLength);
        new Uint8Array(combinedBuffer).set(headerBuffer, 4);
        new Uint8Array(combinedBuffer).set(
          new Uint8Array(arrayBuffer),
          4 + headerBuffer.byteLength
        );
  
        socket.send(combinedBuffer);
  
        console.log(`File ${file.name} sent successfully`);
  
        return new Promise<File>((resolve, reject) => {
          const messageHandler = (event: MessageEvent) => {
            const data = JSON.parse(event.data);
            console.log('mydata', data);
            if (data.type === "file_processed" && data.filename === file.name) {
              socket.removeEventListener("message", messageHandler);
              resolve(file);
            } else if (data.type === "file_upload_error" && data.filename === file.name) {
              console.log('reject')
              socket.removeEventListener("message", messageHandler);
              reject(new Error(`${file.name}`));
            }
          };
          socket.addEventListener("message", messageHandler);
        });
      } catch (error) {
        console.error(`Error sending file ${file.name}:`, error);
        throw new Error(`Failed to upload ${file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    });
  
    try {
      const results = await Promise.allSettled(filePromises);
      
      const successfulFiles = results
        .filter((result): result is PromiseFulfilledResult<File> => result.status === 'fulfilled')
        .map(result => result.value);
  
      const failedFiles = results
        .filter((result): result is PromiseRejectedResult => result.status === 'rejected')
        .map(result => result.reason.message);
  
      if (failedFiles.length > 0) {
        console.error("Some files failed to process:", failedFiles);
        
        // Create an alert message with the names of failed files
        const alertMessage = `The following files failed to upload:\n${failedFiles.join('\n')}`;
        
        // Show the alert to the user
        alert(alertMessage);
      }
  
      const newProcessedFiles = [...filesAttached, ...successfulFiles];
      setFilesAttached(newProcessedFiles);
    } catch (error) {
      console.error("Error processing files:", error);
    }
  };

  return (
    <div className="container my-4">
      {!chatOnly && (
        <IconButton
          onClick={() => setIsLeftPanelOpen(!isLeftPanelOpen)}
          style={{ position: "absolute", left: 10, top: 10, zIndex: 1000 }}
        >
          <MenuIcon />
        </IconButton>
      )}
      <LeftPanel
        isOpen={isLeftPanelOpen}
        onClose={() => setIsLeftPanelOpen(false)}
        libraries={libraries}
        onLibrarySelect={handleLibrarySelect}
        onCreateNewLibrary={() => setIsCreateLibraryModalOpen(true)}
        selectedLibrary={selectedLibrary}
        onDeleteLibrary={handleDeleteLibrary}
      />
      {!chatOnly && (
        <CreateLibraryModal
          isOpen={isCreateLibraryModalOpen}
          onClose={() => setIsCreateLibraryModalOpen(false)}
          onCreateLibrary={handleCreateNewLibrary}
        />
      )}
      <Grid
        container
        justifyContent="center" // Centers horizontally in the available space
        alignItems="center" // Centers vertically
        sx={{ height: "100vh", width: "100%" }} // Ensures the Grid takes full viewport height and width
      >
        <Grid
          sx={{
            backgroundColor: "#f5f5f5",
            padding: "20px",
            borderRadius: "10px",
          }}
          item
          xs={12}
          sm={8}
        >
          <ChatLayout
            messages={messages}
            onSendMessage={handleSendMessage}
            extend={extend}
            onLogout={handleLogout}
            chatOnly={chatOnly}
            onFileAttach={handleFileAttach}
          />

          <PDFDrawer
            isOpen={isDrawerOpen}
            onClose={toggleDrawer(false)}
            library_name="LEX"
          />
          <ErrorModal
            isOpen={isErrorModalOpen}
            onClose={() => setIsErrorModalOpen(false)}
            errorMessage={error}
          />
        </Grid>
      </Grid>
    </div>
  );
};

export default Main;
