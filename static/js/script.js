// Custom JavaScript

// Show a loading message when the form is submitted
document.addEventListener('DOMContentLoaded', function() {
    var fileForm = document.getElementById('file-form');
    if (fileForm) {
        fileForm.addEventListener('submit', function() {
            var submitButton = document.getElementById('file-submit-button');
            submitButton.disabled = true;
            submitButton.innerHTML = 'Processing...';
        });
    }
});
