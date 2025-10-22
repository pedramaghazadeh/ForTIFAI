// Mobile Navigation Toggle
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
});

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-link').forEach(n => n.addEventListener('click', () => {
    hamburger.classList.remove('active');
    navMenu.classList.remove('active');
}));

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Navbar background change on scroll
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
});

// Interactive Demo
class PerformanceDemo {
    constructor() {
        this.canvas = document.getElementById('performance-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.slider = document.getElementById('synthetic-ratio');
        this.ratioValue = document.getElementById('ratio-value');
        
        this.init();
    }

    init() {
        if (!this.canvas || !this.slider) return;
        
        this.slider.addEventListener('input', (e) => {
            this.updateRatio(e.target.value);
            this.drawChart();
        });
        
        this.drawChart();
    }

    updateRatio(value) {
        this.ratioValue.textContent = value + '%';
    }

    drawChart() {
        const ratio = parseInt(this.slider.value);
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Clear canvas
        this.ctx.clearRect(0, 0, width, height);
        
        // Set up chart dimensions
        const margin = 40;
        const chartWidth = width - 2 * margin;
        const chartHeight = height - 2 * margin;
        
        // Calculate performance based on synthetic ratio
        const baselinePerformance = this.calculateBaselinePerformance(ratio);
        const fortifaiPerformance = this.calculateFortifaiPerformance(ratio);
        
        // Draw axes
        this.drawAxes(margin, chartWidth, chartHeight);
        
        // Draw performance lines
        this.drawPerformanceLine(baselinePerformance, '#ef4444', 'Baseline', margin, chartWidth, chartHeight);
        this.drawPerformanceLine(fortifaiPerformance, '#10b981', 'ForTIFAI', margin, chartWidth, chartHeight);
        
        // Draw current point
        this.drawCurrentPoint(ratio, baselinePerformance[Math.floor(ratio / 10)], fortifaiPerformance[Math.floor(ratio / 10)], margin, chartWidth, chartHeight);
    }

    drawAxes(margin, chartWidth, chartHeight) {
        this.ctx.strokeStyle = '#e5e7eb';
        this.ctx.lineWidth = 1;
        
        // X-axis
        this.ctx.beginPath();
        this.ctx.moveTo(margin, height - margin);
        this.ctx.lineTo(margin + chartWidth, height - margin);
        this.ctx.stroke();
        
        // Y-axis
        this.ctx.beginPath();
        this.ctx.moveTo(margin, margin);
        this.ctx.lineTo(margin, height - margin);
        this.ctx.stroke();
        
        // X-axis labels
        this.ctx.fillStyle = '#6b7280';
        this.ctx.font = '12px Inter';
        this.ctx.textAlign = 'center';
        for (let i = 0; i <= 10; i++) {
            const x = margin + (i * chartWidth / 10);
            this.ctx.fillText((i * 10) + '%', x, height - margin + 20);
        }
        
        // Y-axis labels
        this.ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const y = height - margin - (i * chartHeight / 5);
            this.ctx.fillText((i * 20) + '%', margin - 10, y + 4);
        }
        
        // Axis titles
        this.ctx.fillStyle = '#374151';
        this.ctx.font = '14px Inter';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('Synthetic Data Ratio', margin + chartWidth / 2, height - 10);
        
        this.ctx.save();
        this.ctx.translate(15, margin + chartHeight / 2);
        this.ctx.rotate(-Math.PI / 2);
        this.ctx.fillText('Model Performance', 0, 0);
        this.ctx.restore();
    }

    drawPerformanceLine(performance, color, label, margin, chartWidth, chartHeight) {
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        
        for (let i = 0; i < performance.length; i++) {
            const x = margin + (i * chartWidth / (performance.length - 1));
            const y = height - margin - (performance[i] * chartHeight / 100);
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.stroke();
        
        // Draw legend
        const legendY = margin + 20 + (label === 'Baseline' ? 0 : 25);
        this.ctx.fillStyle = color;
        this.ctx.fillRect(margin + chartWidth - 120, legendY, 15, 3);
        this.ctx.fillStyle = '#374151';
        this.ctx.font = '12px Inter';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(label, margin + chartWidth - 100, legendY + 8);
    }

    drawCurrentPoint(ratio, baselinePerf, fortifaiPerf, margin, chartWidth, chartHeight) {
        const x = margin + (ratio * chartWidth / 100);
        
        // Baseline point
        const baselineY = height - margin - (baselinePerf * chartHeight / 100);
        this.ctx.fillStyle = '#ef4444';
        this.ctx.beginPath();
        this.ctx.arc(x, baselineY, 6, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // ForTIFAI point
        const fortifaiY = height - margin - (fortifaiPerf * chartHeight / 100);
        this.ctx.fillStyle = '#10b981';
        this.ctx.beginPath();
        this.ctx.arc(x, fortifaiY, 6, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Performance values
        this.ctx.fillStyle = '#374151';
        this.ctx.font = '11px Inter';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(baselinePerf.toFixed(1) + '%', x, baselineY - 10);
        this.ctx.fillText(fortifaiPerf.toFixed(1) + '%', x, fortifaiY - 10);
    }

    calculateBaselinePerformance(ratio) {
        // Simulate baseline performance degradation
        const performance = [];
        for (let i = 0; i <= 100; i += 10) {
            if (i <= 30) {
                performance.push(100 - i * 0.5);
            } else if (i <= 60) {
                performance.push(85 - (i - 30) * 1.2);
            } else {
                performance.push(Math.max(20, 49 - (i - 60) * 0.8));
            }
        }
        return performance;
    }

    calculateFortifaiPerformance(ratio) {
        // Simulate ForTIFAI performance (more resilient)
        const performance = [];
        for (let i = 0; i <= 100; i += 10) {
            if (i <= 50) {
                performance.push(100 - i * 0.2);
            } else if (i <= 80) {
                performance.push(90 - (i - 50) * 0.8);
            } else {
                performance.push(Math.max(40, 66 - (i - 80) * 0.5));
            }
        }
        return performance;
    }
}

// Initialize demo when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PerformanceDemo();
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-up');
        }
    });
}, observerOptions);

// Observe elements for animation
document.addEventListener('DOMContentLoaded', () => {
    const animateElements = document.querySelectorAll('.result-card, .code-card, .author-card, .step');
    animateElements.forEach(el => observer.observe(el));
});

// Model collapse diagram animation
document.addEventListener('DOMContentLoaded', () => {
    const modelStages = document.querySelectorAll('.model-stage');
    
    modelStages.forEach((stage, index) => {
        stage.addEventListener('mouseenter', () => {
            const performanceBar = stage.querySelector('.performance-fill');
            const currentWidth = performanceBar.style.width;
            
            // Animate the performance bar
            performanceBar.style.transition = 'width 0.5s ease';
            performanceBar.style.width = '100%';
            
            setTimeout(() => {
                performanceBar.style.width = currentWidth;
            }, 1000);
        });
    });
});

// Smooth reveal animation for hero section
document.addEventListener('DOMContentLoaded', () => {
    const heroElements = document.querySelectorAll('.hero-title, .hero-subtitle, .hero-description, .hero-buttons');
    
    heroElements.forEach((element, index) => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        
        setTimeout(() => {
            element.style.transition = 'all 0.6s ease';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, index * 200);
    });
});

// Performance optimization: Debounce scroll events
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Apply debounced scroll handler
const debouncedScrollHandler = debounce(() => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
}, 10);

window.addEventListener('scroll', debouncedScrollHandler);

// Add loading states for interactive elements
document.addEventListener('DOMContentLoaded', () => {
    const interactiveElements = document.querySelectorAll('.btn, .code-link, .nav-link');
    
    interactiveElements.forEach(element => {
        element.addEventListener('click', function() {
            if (this.classList.contains('btn')) {
                this.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    this.style.transform = 'scale(1)';
                }, 150);
            }
        });
    });
});

// Console welcome message
console.log(`
🚀 ForTIFAI Website Loaded Successfully!

📄 Paper: https://arxiv.org/abs/2509.08972
💻 Code: https://github.com/your-username/ForTIFAI

Built with ❤️ for the AI research community.
`);
