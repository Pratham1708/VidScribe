document.addEventListener('DOMContentLoaded', function() {
    // Summary generation
    const summarizeBtn = document.getElementById('summarize-btn');
    const summaryTab = document.getElementById('summary-tab');
    const summaryLoading = document.getElementById('summary-loading');
    const summaryResult = document.getElementById('summary-result');
    
    summarizeBtn.addEventListener('click', function() {
        // Switch to summary tab
        const tabTrigger = new bootstrap.Tab(summaryTab);
        tabTrigger.show();
        
        // Show loading indicator
        summaryLoading.classList.remove('d-none');
        summaryResult.innerHTML = '';
        
        // Request summary
        fetch('/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                summaryResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                summaryLoading.classList.add('d-none');
                return;
            }
            
            // Poll for results
            pollForResults('/summary_result/' + data.task_id, function(result) {
                summaryLoading.classList.add('d-none');
                if (result.error) {
                    summaryResult.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
                } else {
                    summaryResult.innerHTML = `<div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Summary</h5>
                            <p class="card-text">${result.summary}</p>
                        </div>
                    </div>`;
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            summaryLoading.classList.add('d-none');
            summaryResult.innerHTML = `<div class="alert alert-danger">An error occurred while generating the summary.</div>`;
        });
    });
    
    // Question answering
    const askBtn = document.getElementById('ask-btn');
    const questionInput = document.getElementById('question-input');
    const qaTab = document.getElementById('qa-tab');
    const qaLoading = document.getElementById('qa-loading');
    const qaResult = document.getElementById('qa-result');
    
    askBtn.addEventListener('click', function() {
        const question = questionInput.value.trim();
        if (question === '') return;
        
        // Show loading indicator
        qaLoading.classList.remove('d-none');
        qaResult.innerHTML = '';
        
        // Request answer
        fetch('/ask_question', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                qaResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                qaLoading.classList.add('d-none');
                return;
            }
            
            // Poll for results
            pollForResults('/question_result/' + data.task_id, function(result) {
                qaLoading.classList.add('d-none');
                if (result.error) {
                    qaResult.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
                } else {
                    qaResult.innerHTML = `<div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Question: ${question}</h5>
                            <p class="card-text">${result.answer}</p>
                        </div>
                    </div>`;
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            qaLoading.classList.add('d-none');
            qaResult.innerHTML = `<div class="alert alert-danger">An error occurred while processing your question.</div>`;
        });
    });
    
    // Sentiment analysis
    const sentimentBtn = document.getElementById('sentiment-btn');
    const analysisTab = document.getElementById('analysis-tab');
    const sentimentPillTab = document.getElementById('sentiment-pill-tab');
    const sentimentLoading = document.getElementById('sentiment-loading');
    const sentimentResult = document.getElementById('sentiment-result');
    
    sentimentBtn.addEventListener('click', function() {
        // Switch to analysis tab and sentiment pill
        const tabTrigger = new bootstrap.Tab(analysisTab);
        tabTrigger.show();
        const pillTrigger = new bootstrap.Tab(sentimentPillTab);
        pillTrigger.show();
        
        // Show loading indicator
        sentimentLoading.classList.remove('d-none');
        sentimentResult.innerHTML = '';
        
        // Request sentiment analysis
        fetch('/analyze_sentiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                sentimentResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                sentimentLoading.classList.add('d-none');
                return;
            }
            
            // Poll for results
            pollForResults('/sentiment_result/' + data.task_id, function(result) {
                sentimentLoading.classList.add('d-none');
                if (result.error) {
                    sentimentResult.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
                } else {
                    let positiveCount = 0;
                    let negativeCount = 0;
                    
                    let html = '<div class="card"><div class="card-body">';
                    html += '<h5 class="card-title">Sentiment Analysis Results</h5>';
                    
                    // Add results table
                    html += '<div class="table-responsive"><table class="table table-striped">';
                    html += '<thead><tr><th>Segment</th><th>Sentiment</th><th>Confidence</th></tr></thead>';
                    html += '<tbody>';
                    
                    result.results.forEach((item, index) => {
                        if (item.label === 'POSITIVE') positiveCount++;
                        else negativeCount++;
                        
                        html += `<tr>
                            <td>${item.sentence}</td>
                            <td class="sentiment-${item.label.toLowerCase()}">${item.label}</td>
                            <td>${(item.score * 100).toFixed(1)}%</td>
                        </tr>`;
                    });
                    
                    html += '</tbody></table></div>';
                    
                    // Add summary
                    const total = result.results.length;
                    html += '<div class="mt-4"><h6>Summary</h6>';
                    html += `<p>Total segments analyzed: ${total}</p>`;
                    html += `<p class="sentiment-positive">Positive segments: ${positiveCount} (${(positiveCount/total*100).toFixed(1)}%)</p>`;
                    html += `<p class="sentiment-negative">Negative segments: ${negativeCount} (${(negativeCount/total*100).toFixed(1)}%)</p>`;
                    html += '</div>';
                    
                    html += '</div></div>';
                    sentimentResult.innerHTML = html;
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            sentimentLoading.classList.add('d-none');
            sentimentResult.innerHTML = `<div class="alert alert-danger">An error occurred while analyzing sentiment.</div>`;
        });
    });
    
    // Word cloud generation
    const wordcloudBtn = document.getElementById('wordcloud-btn');
    const wordcloudPillTab = document.getElementById('wordcloud-pill-tab');
    const wordcloudLoading = document.getElementById('wordcloud-loading');
    const wordcloudResult = document.getElementById('wordcloud-result');
    
    wordcloudBtn.addEventListener('click', function() {
        // Switch to analysis tab and wordcloud pill
        const tabTrigger = new bootstrap.Tab(analysisTab);
        tabTrigger.show();
        const pillTrigger = new bootstrap.Tab(wordcloudPillTab);
        pillTrigger.show();
        
        // Show loading indicator
        wordcloudLoading.classList.remove('d-none');
        wordcloudResult.innerHTML = '';
        
        // Request word cloud
        fetch('/generate_wordcloud', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                wordcloudResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                wordcloudLoading.classList.add('d-none');
                return;
            }
            
            // Poll for results
            pollForResults('/wordcloud_result/' + data.task_id, function(result) {
                wordcloudLoading.classList.add('d-none');
                if (result.error) {
                    wordcloudResult.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
                } else {
                    wordcloudResult.innerHTML = `
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Word Cloud</h5>
                                <img src="data:image/png;base64,${result.image_data}" alt="Word Cloud" class="img-fluid">
                            </div>
                        </div>`;
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            wordcloudLoading.classList.add('d-none');
            wordcloudResult.innerHTML = `<div class="alert alert-danger">An error occurred while generating the word cloud.</div>`;
        });
    });
    
    // Transcript comparison
    const compareBtn = document.getElementById('compare-btn');
    const comparisonModal = new bootstrap.Modal(document.getElementById('comparison-modal'));
    const uploadComparisonBtn = document.getElementById('upload-comparison-btn');
    const comparisonFileModal = document.getElementById('comparison-file-modal');
    const comparisonPillTab = document.getElementById('comparison-pill-tab');
    const comparisonLoading = document.getElementById('comparison-loading');
    const comparisonStatus = document.getElementById('comparison-status');
    const comparisonResult = document.getElementById('comparison-result');
    
    compareBtn.addEventListener('click', function() {
        comparisonModal.show();
    });
    
    uploadComparisonBtn.addEventListener('click', function() {
        if (!comparisonFileModal.files[0]) {
            alert('Please select a file to compare.');
            return;
        }
        
        // Switch to analysis tab and comparison pill
        const tabTrigger = new bootstrap.Tab(analysisTab);
        tabTrigger.show();
        const pillTrigger = new bootstrap.Tab(comparisonPillTab);
        pillTrigger.show();
        
        // Hide modal
        comparisonModal.hide();
        
        // Show loading indicator
        comparisonLoading.classList.remove('d-none');
        comparisonStatus.textContent = 'Transcribing comparison file...';
        comparisonResult.innerHTML = '';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', comparisonFileModal.files[0]);
        
        // Request comparison
        fetch('/compare_transcripts', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                comparisonResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                comparisonLoading.classList.add('d-none');
                return;
            }
            
            // First poll for transcription results
            pollForResults('/comparison_transcription_result/' + data.task_id, function(result) {
                if (result.error) {
                    comparisonLoading.classList.add('d-none');
                    comparisonResult.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
                } else if (result.next_task_id) {
                    // Now poll for comparison results
                    comparisonStatus.textContent = 'Computing similarity...';
                    
                    // Show preview of second transcript
                    comparisonResult.innerHTML = `
                        <div class="card mb-3">
                            <div class="card-body">
                                <h6>Preview of comparison transcript:</h6>
                                <p>${result.transcript2}</p>
                            </div>
                        </div>`;
                    
                    pollForResults('/comparison_result/' + result.next_task_id, function(compResult) {
                        comparisonLoading.classList.add('d-none');
                        
                        if (compResult.error) {
                            comparisonResult.innerHTML += `<div class="alert alert-danger">${compResult.error}</div>`;
                        } else {
                            let similarityClass = '';
                            if (compResult.similarity > 0.8) similarityClass = 'text-success';
                            else if (compResult.similarity > 0.5) similarityClass = 'text-warning';
                            else similarityClass = 'text-danger';
                            
                            comparisonResult.innerHTML += `
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Comparison Results</h5>
                                        <p>Semantic similarity: <span class="${similarityClass}">${(compResult.similarity * 100).toFixed(1)}%</span></p>
                                        <p>${compResult.interpretation}</p>
                                    </div>
                                </div>`;
                        }
                    });
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            comparisonLoading.classList.add('d-none');
            comparisonResult.innerHTML = `<div class="alert alert-danger">An error occurred during comparison.</div>`;
        });
    });
    
    // Start comparison from the analysis tab
    const startComparisonBtn = document.getElementById('start-comparison-btn');
    const comparisonFile = document.getElementById('comparison-file');
    
    startComparisonBtn.addEventListener('click', function() {
        if (!comparisonFile.files[0]) {
            alert('Please select a file to compare.');
            return;
        }
        
        // Show loading indicator
        comparisonLoading.classList.remove('d-none');
        comparisonStatus.textContent = 'Transcribing comparison file...';
        comparisonResult.innerHTML = '';
        
        // Create form data
        const formData = new FormData();
        formData.append('file', comparisonFile.files[0]);
        
        // Request comparison
        fetch('/compare_transcripts', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                comparisonResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                comparisonLoading.classList.add('d-none');
                return;
            }
            
            // First poll for transcription results
            pollForResults('/comparison_transcription_result/' + data.task_id, function(result) {
                if (result.error) {
                    comparisonLoading.classList.add('d-none');
                    comparisonResult.innerHTML = `<div class="alert alert-danger">${result.error}</div>`;
                } else if (result.next_task_id) {
                    // Now poll for comparison results
                    comparisonStatus.textContent = 'Computing similarity...';
                    
                    // Show preview of second transcript
                    comparisonResult.innerHTML = `
                        <div class="card mb-3">
                            <div class="card-body">
                                <h6>Preview of comparison transcript:</h6>
                                <p>${result.transcript2}</p>
                            </div>
                        </div>`;
                    
                    pollForResults('/comparison_result/' + result.next_task_id, function(compResult) {
                        comparisonLoading.classList.add('d-none');
                        
                        if (compResult.error) {
                            comparisonResult.innerHTML += `<div class="alert alert-danger">${compResult.error}</div>`;
                        } else {
                            let similarityClass = '';
                            if (compResult.similarity > 0.8) similarityClass = 'text-success';
                            else if (compResult.similarity > 0.5) similarityClass = 'text-warning';
                            else similarityClass = 'text-danger';
                            
                            comparisonResult.innerHTML += `
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Comparison Results</h5>
                                        <p>Semantic similarity: <span class="${similarityClass}">${(compResult.similarity * 100).toFixed(1)}%</span></p>
                                        <p>${compResult.interpretation}</p>
                                    </div>
                                </div>`;
                        }
                    });
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            comparisonLoading.classList.add('d-none');
            comparisonResult.innerHTML = `<div class="alert alert-danger">An error occurred during comparison.</div>`;
        });
    });
    
    // Helper function to poll for results
    function pollForResults(url, callback) {
        const poll = function() {
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'completed') {
                        callback(data);
                    } else if (data.status === 'error') {
                        callback({ error: data.error });
                    } else {
                        // Still processing, poll again after a delay
                        setTimeout(poll, 1000);
                    }
                })
                .catch(error => {
                    console.error('Polling error:', error);
                    callback({ error: 'An error occurred while retrieving results.' });
                });
        };
        
        poll();
    }
});