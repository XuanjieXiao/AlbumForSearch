// AlbumForSearch/static/script.js
document.addEventListener('DOMContentLoaded', () => {
    // Nav Upload Elements
    const navUploadButton = document.getElementById('nav-upload-button');
    const hiddenUploadInput = document.getElementById('unified-upload-input-hidden');
    const mainUploadStatus = document.getElementById('upload-status-main'); 

    // Search Elements
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchStatus = document.getElementById('search-status');
    
    // Gallery Elements
    const imageGallery = document.getElementById('image-gallery');
    const loadingGallery = document.getElementById('loading-gallery');
    const totalImagesCountSpan = document.getElementById('total-images-count');
    const searchResultsTitleSpan = document.getElementById('search-results-title');
    const noMoreResultsDiv = document.getElementById('no-more-results');

    // Modal elements
    const modal = document.getElementById('image-modal');
    const modalImageElement = document.getElementById('modal-image-element');
    const modalFilename = document.getElementById('modal-filename');
    const modalSimilarity = document.getElementById('modal-similarity');
    const modalSimilarityContainer = document.getElementById('modal-similarity-container');
    const modalQwenDescription = document.getElementById('modal-qwen-description');
    const modalQwenKeywords = document.getElementById('modal-qwen-keywords');
    const modalIsEnhanced = document.getElementById('modal-is-enhanced');
    const modalEnhanceButton = document.getElementById('modal-enhance-button');
    const closeModalButton = document.querySelector('.close-button');
    let currentModalImageId = null;
    let currentModalIsSearchResult = false;

    // Pagination elements
    const paginationControls = document.querySelector('.pagination-controls');
    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const currentPageInfoSpan = document.getElementById('current-page-info');
    let galleryCurrentPage = 1;
    const galleryImagesPerPage = 20;
    let galleryTotalPages = 1;

    // Search results and infinite scroll state
    let currentSearchResults = [];
    let displayedSearchResultsCount = 0;
    const searchResultsBatchSize = 20;
    // Default thresholds, will be updated by search response
    let ENHANCED_SEARCH_THRESHOLD = 0.50; // Keep this as a reference or default
    let CLIP_ONLY_SEARCH_THRESHOLD = 0.19; // Your new threshold for CLIP-only
    let isLoadingMoreSearchResults = false;
    let navUploadAbortController = null; 

    // --- Event Listeners for Main Page (index.html) ---
    if (navUploadButton && hiddenUploadInput) {
        navUploadButton.addEventListener('click', () => {
            hiddenUploadInput.value = null; 
            hiddenUploadInput.click(); 
        });
        hiddenUploadInput.addEventListener('change', () => {
            if (hiddenUploadInput.files.length > 0) {
                handleUnifiedUpload(hiddenUploadInput.files, navUploadButton, hiddenUploadInput, mainUploadStatus);
            }
        });
    }
    
    if (searchButton) {
        searchButton.addEventListener('click', performSearch);
    }
    if (searchInput) {
        searchInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                performSearch();
            }
        });
    }

    function handleUnifiedUpload(files, buttonElement, inputElement, statusElement) {
        if (!files || files.length === 0) {
            if(statusElement) statusElement.textContent = '请先选择文件。';
            return;
        }
        if (navUploadAbortController) { 
            navUploadAbortController.abort();
        }
        navUploadAbortController = new AbortController();
        const signal = navUploadAbortController.signal;

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            if (files[i].type && !files[i].type.startsWith('image/')) {
                 console.warn(`(Main Page Upload) 跳过非图片文件: ${files[i].name} (type: ${files[i].type})`);
                 continue;
            }
            formData.append('files', files[i]);
        }
        const fileCount = formData.getAll('files').length;
        if (fileCount === 0) {
            if(statusElement) statusElement.textContent = '选择的文件中没有有效的图片文件。';
            if (inputElement) inputElement.value = null;
            return;
        }

        if(statusElement) statusElement.textContent = `正在上传 ${fileCount} 张图片...`;
        if (buttonElement) buttonElement.disabled = true;

        fetch('/upload_images', {
            method: 'POST',
            body: formData,
            signal: signal
        })
        .then(response => response.json())
        .then(data => {
            if (signal.aborted) return; 
            if (data.error) {
                if(statusElement) statusElement.textContent = `上传失败: ${data.error}`;
            } else {
                if(statusElement) statusElement.textContent = data.message || `成功处理 ${data.processed_files?.length || 0} 张图片。`;
                switchToGalleryView(); 
            }
        })
        .catch(error => {
            if (error.name === 'AbortError') {
                if(statusElement) statusElement.textContent = '上传已取消。';
            } else {
                console.error('上传错误 (Main Page):', error);
                if(statusElement) statusElement.textContent = '上传过程中发生网络错误。';
            }
        })
        .finally(() => {
            if (buttonElement) buttonElement.disabled = false;
            if (inputElement) inputElement.value = null;
            navUploadAbortController = null;
            setTimeout(() => {
                if (statusElement && (statusElement.textContent.includes("上传") || statusElement.textContent.includes("处理"))) {
                   // statusElement.textContent = ''; // Optional: auto-clear
                }
            }, 7000);
        });
    }

    function performSearch() {
        const queryText = searchInput.value.trim();
        if (!queryText) {
            if(searchStatus) searchStatus.textContent = '请输入搜索描述。';
            return;
        }
        imageGallery.innerHTML = '';
        loadingGallery.style.display = 'flex'; 
        if(searchButton) searchButton.disabled = true;
        if(searchStatus) searchStatus.textContent = '正在搜索...';
        if(mainUploadStatus) mainUploadStatus.textContent = '';
        if(paginationControls) paginationControls.style.display = 'none';
        if(searchResultsTitleSpan) searchResultsTitleSpan.style.display = 'inline';
        if(noMoreResultsDiv) noMoreResultsDiv.style.display = 'none';

        fetch('/search_images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query_text: queryText, top_k: 200 })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                if(searchStatus) searchStatus.textContent = `搜索失败: ${data.error}`;
                imageGallery.innerHTML = `<p>搜索失败: ${data.error}</p>`;
                currentSearchResults = [];
            } else {
                // Determine threshold based on backend search mode
                const activeSimilarityThreshold = data.search_mode_is_enhanced ? ENHANCED_SEARCH_THRESHOLD : CLIP_ONLY_SEARCH_THRESHOLD;
                
                currentSearchResults = data.results.filter(img => img.similarity >= activeSimilarityThreshold);
                
                if (currentSearchResults.length > 0) {
                    if(searchStatus) searchStatus.textContent = `找到 ${currentSearchResults.length} 张相似度 >= ${activeSimilarityThreshold.toFixed(2)} 的相关图片。`;
                    displayedSearchResultsCount = 0;
                    imageGallery.innerHTML = '';
                    loadMoreSearchResults();
                } else {
                    if(searchStatus) searchStatus.textContent = `未找到相似度 >= ${activeSimilarityThreshold.toFixed(2)} 的图片。`;
                    imageGallery.innerHTML = `<p>未找到与描述 "${queryText}" 匹配且相似度足够高的图片 (阈值: ${activeSimilarityThreshold.toFixed(2)})。</p>`;
                    currentSearchResults = [];
                }
            }
        })
        .catch(error => {
            console.error('搜索错误:', error);
            if(searchStatus) searchStatus.textContent = '搜索过程中发生网络错误。';
            imageGallery.innerHTML = '<p>搜索过程中发生网络错误。</p>';
            currentSearchResults = [];
        })
        .finally(() => {
            loadingGallery.style.display = 'none';
            if(searchButton) searchButton.disabled = false;
        });
    }

    function loadMoreSearchResults() {
        if (isLoadingMoreSearchResults) return;
        isLoadingMoreSearchResults = true;
        loadingGallery.style.display = 'flex';

        const nextBatch = currentSearchResults.slice(
            displayedSearchResultsCount,
            displayedSearchResultsCount + searchResultsBatchSize
        );

        if (nextBatch.length > 0) {
            displayImages(nextBatch, true, true); 
            displayedSearchResultsCount += nextBatch.length;
            if(noMoreResultsDiv) noMoreResultsDiv.style.display = 'none';
        } else {
            if (displayedSearchResultsCount > 0 && currentSearchResults.length > 0 && displayedSearchResultsCount >= currentSearchResults.length) {
                 if(noMoreResultsDiv) noMoreResultsDiv.style.display = 'block';
            } else if (displayedSearchResultsCount === 0 && currentSearchResults.length === 0) {
                if(noMoreResultsDiv) noMoreResultsDiv.style.display = 'none';
            }
        }
        isLoadingMoreSearchResults = false;
        loadingGallery.style.display = 'none';
    }

    function displayImages(images, isSearchResult = false, append = false) {
        if (!append) {
            imageGallery.innerHTML = '';
        }
        if (!images || images.length === 0) {
            if (!append) { 
                 // Determine threshold based on the context if possible, or use a general message
                 const activeThreshold = isSearchResult ? (document.getElementById('use-enhanced-search-toggle-controls') && !document.getElementById('use-enhanced-search-toggle-controls').checked ? CLIP_ONLY_SEARCH_THRESHOLD : ENHANCED_SEARCH_THRESHOLD) : ENHANCED_SEARCH_THRESHOLD;
                 if (isSearchResult) imageGallery.innerHTML = `<p>未找到相似度 >= ${activeThreshold.toFixed(2)} 的图片。</p>`;
                 else imageGallery.innerHTML = '<p>图片库为空，请上传图片。</p>';
            }
            return;
        }
        images.forEach(img => {
            const item = document.createElement('div');
            item.classList.add('gallery-item');
            item.dataset.imageId = img.id;
            item.dataset.originalUrl = img.original_url;
            item.dataset.filename = img.filename;
            if (isSearchResult && img.similarity !== undefined) {
                item.dataset.similarity = img.similarity.toFixed(4);
            }
            const imgElement = document.createElement('img');
            imgElement.src = img.thumbnail_url || 'https://placehold.co/160x130/eee/ccc?text=NoThumb';
            imgElement.alt = img.filename;
            imgElement.onerror = () => { imgElement.src = 'https://placehold.co/160x130/eee/ccc?text=Error'; };
            const nameElement = document.createElement('p');
            nameElement.textContent = img.filename.length > 20 ? img.filename.substring(0, 17) + '...' : img.filename;
            item.appendChild(imgElement);
            item.appendChild(nameElement);
            if (isSearchResult && img.similarity !== undefined) {
                const similarityElement = document.createElement('p');
                similarityElement.classList.add('similarity');
                similarityElement.textContent = `相似度: ${img.similarity.toFixed(4)}`;
                item.appendChild(similarityElement);
            }
            if (img.is_enhanced) {
                const enhancedBadge = document.createElement('span');
                enhancedBadge.classList.add('enhanced-badge');
                enhancedBadge.textContent = '已增强';
                item.appendChild(enhancedBadge);
            }
            item.addEventListener('click', () => {
                openModal(img.id, img.original_url, img.filename, isSearchResult, item.dataset.similarity);
            });
            imageGallery.appendChild(item);
        });
    }

    function openModal(imageId, originalUrl, filename, isSearchResult, similarityScore) {
        currentModalImageId = imageId;
        currentModalIsSearchResult = isSearchResult;
        modalImageElement.src = originalUrl || '';
        modalFilename.textContent = filename;
        if (isSearchResult && similarityScore !== undefined && similarityScore !== 'N/A') {
            modalSimilarity.textContent = similarityScore;
            modalSimilarityContainer.style.display = 'block';
        } else {
            modalSimilarity.textContent = 'N/A';
            modalSimilarityContainer.style.display = 'none';
        }
        modalQwenDescription.textContent = '加载中...';
        modalQwenKeywords.textContent = '加载中...';
        modalIsEnhanced.textContent = '加载中...';
        modalEnhanceButton.style.display = 'none';
        modal.style.display = 'block';

        fetch(`/image_details/${imageId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    modalQwenDescription.textContent = '获取详情失败';
                    modalQwenKeywords.textContent = '获取详情失败';
                    modalIsEnhanced.textContent = '获取详情失败';
                    return;
                }
                modalQwenDescription.textContent = data.qwen_description || '无';
                modalQwenKeywords.textContent = data.qwen_keywords && data.qwen_keywords.length > 0 ? data.qwen_keywords.join(', ') : '无';
                const isEnhanced = data.is_enhanced;
                modalIsEnhanced.textContent = isEnhanced ? '是' : '否';
                if (!isEnhanced) {
                    modalEnhanceButton.style.display = 'inline-block';
                } else {
                    modalEnhanceButton.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('获取图片详情API错误:', error);
                modalQwenDescription.textContent = '网络错误';
                modalQwenKeywords.textContent = '网络错误';
                modalIsEnhanced.textContent = '网络错误';
            });
    }

    if (closeModalButton) {
        closeModalButton.addEventListener('click', () => {
            modal.style.display = 'none';
            currentModalImageId = null;
        });
    }
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
            currentModalImageId = null;
        }
    });

    if (modalEnhanceButton) {
        modalEnhanceButton.addEventListener('click', () => {
            if (!currentModalImageId) return;
            modalEnhanceButton.disabled = true;
            modalEnhanceButton.textContent = '正在增强...';
            fetch(`/enhance_image/${currentModalImageId}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert(`增强失败: ${data.error}`);
                        if(data.is_enhanced !== undefined) { 
                            modalIsEnhanced.textContent = data.is_enhanced ? '是' : '否';
                            if (data.qwen_description) modalQwenDescription.textContent = data.qwen_description;
                            if (data.qwen_keywords) modalQwenKeywords.textContent = data.qwen_keywords.join(', ');
                            if(data.is_enhanced) modalEnhanceButton.style.display = 'none';
                        }
                    } else {
                        alert(data.message || '增强请求已发送，图片已更新。');
                        modalQwenDescription.textContent = data.qwen_description || '无';
                        modalQwenKeywords.textContent = data.qwen_keywords && data.qwen_keywords.length > 0 ? data.qwen_keywords.join(', ') : '无';
                        modalIsEnhanced.textContent = data.is_enhanced ? '是' : '否';
                        if (data.is_enhanced) {
                            modalEnhanceButton.style.display = 'none';
                        }
                        const galleryItem = document.querySelector(`.gallery-item[data-image-id='${currentModalImageId}']`);
                        if (galleryItem) {
                            galleryItem.dataset.isEnhanced = data.is_enhanced.toString();
                            const existingBadge = galleryItem.querySelector('.enhanced-badge');
                            if (data.is_enhanced && !existingBadge) {
                                 const enhancedBadge = document.createElement('span');
                                 enhancedBadge.classList.add('enhanced-badge');
                                 enhancedBadge.textContent = '已增强';
                                 galleryItem.appendChild(enhancedBadge);
                            } else if (!data.is_enhanced && existingBadge) {
                                existingBadge.remove();
                            }
                        }
                    }
                })
                .catch(error => {
                    console.error('增强图片API错误:', error);
                    alert('增强请求失败。');
                })
                .finally(() => {
                    modalEnhanceButton.disabled = false;
                    modalEnhanceButton.textContent = '对此图片进行增强分析';
                });
        });
    }

    function updateGalleryPaginationControls(totalImages, currentPageNum, imagesPerPageNum, totalPagesCalculated) {
        galleryTotalPages = totalPagesCalculated > 0 ? totalPagesCalculated : 1;
        totalImagesCountSpan.textContent = totalImages;
        currentPageInfoSpan.textContent = `第 ${currentPageNum} / ${galleryTotalPages} 页`;
        prevPageButton.disabled = currentPageNum === 1;
        nextPageButton.disabled = currentPageNum === galleryTotalPages || totalImages === 0;
    }
    
    if (prevPageButton) {
        prevPageButton.addEventListener('click', () => {
            if (galleryCurrentPage > 1) {
                loadGalleryImages(galleryCurrentPage - 1);
            }
        });
    }
    if (nextPageButton) {
        nextPageButton.addEventListener('click', () => {
            if (galleryCurrentPage < galleryTotalPages) {
                loadGalleryImages(galleryCurrentPage + 1);
            }
        });
    }

    function switchToGalleryView() {
        currentSearchResults = [];
        displayedSearchResultsCount = 0;
        imageGallery.innerHTML = '';
        if(searchInput) searchInput.value = '';
        if(searchStatus) searchStatus.textContent = '';
        if(mainUploadStatus) mainUploadStatus.textContent = ''; 
        if(searchResultsTitleSpan) searchResultsTitleSpan.style.display = 'none';
        if(paginationControls) paginationControls.style.display = 'block';
        if(noMoreResultsDiv) noMoreResultsDiv.style.display = 'none';
        loadGalleryImages(1);
    }

    function loadGalleryImages(page = 1) {
        galleryCurrentPage = page;
        imageGallery.innerHTML = '';
        loadingGallery.style.display = 'flex';
        if(searchStatus) searchStatus.textContent = '';
        if(searchResultsTitleSpan) searchResultsTitleSpan.style.display = 'none';
        if(paginationControls) paginationControls.style.display = 'block';
        if(noMoreResultsDiv) noMoreResultsDiv.style.display = 'none';

        fetch(`/images?page=${galleryCurrentPage}&limit=${galleryImagesPerPage}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    imageGallery.innerHTML = `<p>加载图片库失败: ${data.error}</p>`;
                    updateGalleryPaginationControls(0, 1, galleryImagesPerPage, 1);
                } else {
                    displayImages(data.images, false, false);
                    updateGalleryPaginationControls(data.total_count, data.page, data.limit, data.total_pages);
                    totalImagesCountSpan.textContent = data.total_count;
                }
            })
            .catch(error => {
                console.error('加载图片库错误:', error);
                imageGallery.innerHTML = '<p>加载图片库时发生网络错误。</p>';
                updateGalleryPaginationControls(0, 1, galleryImagesPerPage, 1);
            })
            .finally(() => {
                loadingGallery.style.display = 'none';
            });
    }

    window.addEventListener('scroll', () => {
        if (currentSearchResults.length > 0 && !isLoadingMoreSearchResults) {
            if (paginationControls && paginationControls.style.display === 'none') { 
                 if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 300) {
                    if (displayedSearchResultsCount < currentSearchResults.length) {
                        loadMoreSearchResults();
                    } else if (displayedSearchResultsCount > 0) {
                        noMoreResultsDiv.style.display = 'block';
                    }
                }
            }
        }
    });

    switchToGalleryView(); 
});