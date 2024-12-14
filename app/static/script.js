$(document).ready(function () {
    
    const apiUrl ='https://th-ro-chatbot.onrender.com/chat';  // Production URL
        

    // Function to append messages to the chat window
    function appendMessage(sender, message) {
        const senderClass = sender === 'user' ? 'text-right' : 'text-left';
        const messageColor = sender === 'user' ? '#2EFE2E' : 'white';

        $(".media-list").append(`
            <li class="media">
                <div class="media-body">
                    <div class="media ${senderClass}" style="color: ${messageColor};">
                        <div class="media-body">${message}</div>
                    </div>
                </div>
            </li>
        `);

        // Auto-scroll the chat box to the bottom
        $(".fixed-panel").stop().animate({ scrollTop: $(".fixed-panel")[0].scrollHeight }, 1000);
    }

    // When the "Send" button is clicked
    $('#chatbot-form-btn').click(function (e) {
        e.preventDefault();
        $('#chatbot-form').submit();
    });

    // Handle form submission
    $('#chatbot-form').submit(function (e) {
        e.preventDefault();
        const message = $('#messageText').val().trim();  // Remove unnecessary whitespace

        if (!message) {
            return; // Do nothing if the message is empty
        }

        // Display user message in the chat
        appendMessage('user', 'You: '+message);

        // Send the user's message to the backend API via AJAX
        $.ajax({
            type: "POST",
            url: apiUrl,
            data: JSON.stringify({ message: message }),  // Send message as JSON
            contentType: "application/json",  // Set content type to JSON
            success: function (response) {
                $('#messageText').val('');  // Clear input field after sending
                const answer = response.response;  // Get the chatbot's response

                // Display chatbot's response
                appendMessage('bot', 'TH-Ro: '+answer);
            },
            error: function (error) {
                console.error("Error:", error);
                // Optionally display an error message in the UI
                appendMessage('bot', "Sorry, something went wrong. Please try again.");
            }
        });
    });
});