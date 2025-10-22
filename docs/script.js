// Simple JavaScript for ForTIFAI website
// Minimal functionality for smooth scrolling and basic interactions

// Smooth scrolling for anchor links
document.addEventListener('DOMContentLoaded', function() {
    // Handle smooth scrolling for internal links
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add subtle animation to figure on load
    const figure = document.querySelector('.main-figure');
    if (figure) {
        figure.style.opacity = '0';
        figure.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            figure.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            figure.style.opacity = '1';
            figure.style.transform = 'translateY(0)';
        }, 100);
    }
    
    // Console welcome message
    console.log(`
🚀 ForTIFAI Website Loaded Successfully!

📄 Paper: https://arxiv.org/abs/2509.08972
💻 Code: https://github.com/fortifai/ForTIFAI

Built with ❤️ for the AI research community.
    `);
});