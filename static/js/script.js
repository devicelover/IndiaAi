document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('description-form'); // Target the form
    var descriptionField = document.getElementById('description'); // Target the description field

    if (form && descriptionField) {
        form.addEventListener('submit', function (e) {
            var description = descriptionField.value.trim(); // Get trimmed value
            var wordCount = description.split(/\s+/).filter(word => word).length; // Count words

            // Check for minimum word count
            if (wordCount < 5) {
                e.preventDefault(); // Prevent form submission
                alert('Description must be at least 5 words.\n\nविवरण कम से कम 5 शब्दों का होना चाहिए।');
                return;
            }

            // Check for basic sentence structure
            if (!description.includes(' ')) {
                e.preventDefault();
                alert('Please write a proper sentence.\n\nकृपया सही वाक्य लिखें।');
                return;
            }
        });
    }
});
