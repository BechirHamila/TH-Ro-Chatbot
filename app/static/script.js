$(document).ready(function () {
    const apiUrl = 'https://th-ro-chatbot.onrender.com'; 
    let sessionId = null;

    // Initialize session when the page loads
    initializeSession();

    // initialize a new session
    function initializeSession() {
        $.ajax({
            type: "GET",
            url: `${apiUrl}/new_session`,
            success: function (response) {
                sessionId = response.session_id;
                console.log("New session initialized:", sessionId);
                appendMessage('bot', "TH-Ro: Hello! How can I assist you today?");
            },
            error: function () {
                console.error("Failed to initialize session.");
                appendMessage('bot', "Sorry, something went wrong initializing the chat.");
            }
        });
    }

    // Function to send user input to the backend
    function sendMessage() {
        const userMessage = $("#userInput").val().trim(); 
        if (!userMessage) return; 

        // Display user's message in chat
        appendMessage('user', userMessage);

        // Send the message to the backend API
        $.ajax({
            type: "POST",
            url: `${apiUrl}/chat`,
            data: JSON.stringify({ message: userMessage, session_id: sessionId }),
            contentType: "application/json",
            success: function (response) {
                const botResponse = response.response; // Get the chatbot's response
                appendMessage('bot', botResponse); // Display bot's response
                console.log("Chatbot answer:", botResponse);
            },
            error: function (error) {
                console.error("Error:", error);
                appendMessage('bot', "Sorry, something went wrong. Please try again.");
            }
        });

        $("#userInput").val(""); // Clear the inpuet field aftr sending
    }

    // Function to append messages to the chat window
    function appendMessage(sender, message) {
        const senderClass = sender === 'user' ? 'user-message' : 'bot-message';
        const formattedMessage = `<div class="message ${senderClass}">${message}</div>`;
        $("#chatBody").append(formattedMessage);
        $("#chatBody").scrollTop($("#chatBody")[0].scrollHeight); // Auto-scroll to bottom
    }

    // Event listeners for "Send" button and Enter key
    $("#sendMessage").click(sendMessage);

    $("#userInput").keypress(function (e) {
        if (e.which === 13) { // Enter key triggers send
            sendMessage();
        }
    });
});
