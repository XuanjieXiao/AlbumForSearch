// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const qwenAnalysisToggle = document.getElementById('qwen-analysis-toggle');
    const qwenStatusText = document.getElementById('qwen-status-text');
    // const useEnhancedSearchToggle = document.getElementById('use-enhanced-search-toggle');

    // Unified Upload elements (moved to control panel)
    const unifiedUploadInput = document.getElementById('unified-upload-input');
    const unifiedUploadButton = document.getElementById('unified-upload-button');
    const uploadStatus = document.getElementById('upload-status'); // Still used

    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchStatus = document.getElementById('search-status');
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

    // Pagination elements (for gallery view)
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
    const SIMILARITY_THRESHOLD = 0.5; // Similarity threshold
    let isLoadingMoreSearchResults = false;


    // --- Initial State Setup ---
    fetch('/config/qwen_analysis')
        .then(response => response.json())
        .then(data => {
            qwenAnalysisToggle.checked = data.qwen_vl_analysis_enabled;
            qwenStatusText.textContent = data.qwen_vl_analysis_enabled ? '已开启' : '已关闭';
        })
        .catch(error => console.error('获取Qwen分析配置失败:', error));

    // --- Event Listeners ---
    qwenAnalysisToggle.addEventListener('change', () => {
        const enabled = qwenAnalysisToggle.checked;
        fetch('/config/qwen_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: enabled })
        })
        .then(response => response.json())
        .then(data => {
            qwenStatusText.textContent = data.qwen_vl_analysis_enabled ? '已开启' : '已关闭';
            alert(data.message);
        })
        .catch(error => {
            console.error('更新Qwen分析配置失败:', error);
            alert('更新配置失败');
            qwenAnalysisToggle.checked = !enabled;
            qwenStatusText.textContent = !enabled ? '已开启' : '已关闭';
        });
    });

    // Unified Upload Handler
    unifiedUploadButton.addEventListener('click', () => {
        handleUnifiedUpload();
    });
    // Also allow upload on file selection (optional, can be convenient)
    // unifiedUploadInput.addEventListener('change', handleUnifiedUpload); 

    function handleUnifiedUpload() {
        const files = unifiedUploadInput.files;
        if (!files || files.length === 0) {
            uploadStatus.textContent = '请先选择文件或文件夹。';
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            if (files[i].type && !files[i].type.startsWith('image/')) {
                // Skip non-image files if `accept="image/*"` didn't catch them or for folder uploads
                if (!files[i].type && files[i].name) { // Heuristic for files from folder upload without type
                    const lcFilename = files[i].name.toLowerCase();
                    if (!lcFilename.endsWith('.jpg') && !lcFilename.endsWith('.jpeg') &&
                        !lcFilename.endsWith('.png') && !lcFilename.endsWith('.gif') &&
                        !lcFilename.endsWith('.bmp') && !lcFilename.endsWith('.webp')) {
                        console.warn(`根据文件名跳过疑似非图片文件: ${files[i].name}`);
                        continue;
                    }
                } else if (files[i].type && !files[i].type.startsWith('image/')) {
                     console.warn(`跳过非图片文件: ${files[i].name} (type: ${files[i].type})`);
                     continue;
                }
            }
            formData.append('files', files[i]);
        }

        const fileCount = formData.getAll('files').length;
        if (fileCount === 0) {
            uploadStatus.textContent = '选择的文件中没有有效的图片文件。';
            unifiedUploadInput.value = null; // Clear file input
            return;
        }

        uploadStatus.textContent = `正在上传 ${fileCount} 张图片...`;
        unifiedUploadButton.disabled = true;
        // TODO: Implement progress bar if needed

        fetch('/upload_images', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                uploadStatus.textContent = `上传失败: ${data.error}`;
            } else {
                uploadStatus.textContent = data.message || `成功处理 ${data.processed_files?.length || 0} 张图片。`;
                switchToGalleryView(); // Switch to gallery view and refresh
            }
        })
        .catch(error => {
            console.error('上传错误:', error);
            uploadStatus.textContent = '上传过程中发生网络错误。';
        })
        .finally(() => {
            unifiedUploadButton.disabled = false;
            unifiedUploadInput.value = null; // Clear file input
            // TODO: Hide progress bar
        });
    }

    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            performSearch();
        }
    });

    function performSearch() {
        const queryText = searchInput.value.trim();
        if (!queryText) {
            searchStatus.textContent = '请输入搜索描述。';
            return;
        }

        imageGallery.innerHTML = ''; // Clear previous results/gallery
        loadingGallery.style.display = 'block';
        searchButton.disabled = true;
        searchStatus.textContent = '正在搜索...';
        paginationControls.style.display = 'none'; // Hide pagination for search
        searchResultsTitleSpan.style.display = 'inline';
        noMoreResultsDiv.style.display = 'none';


        // Request a larger top_k from backend, e.g., 200
        fetch('/search_images', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query_text: queryText, top_k: 200 }) // Request more results
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                searchStatus.textContent = `搜索失败: ${data.error}`;
                imageGallery.innerHTML = `<p>搜索失败: ${data.error}</p>`;
                currentSearchResults = [];
            } else {
                // Filter results by similarity client-side
                currentSearchResults = data.results.filter(img => img.similarity >= SIMILARITY_THRESHOLD);
                
                if (currentSearchResults.length > 0) {
                    searchStatus.textContent = `找到 ${currentSearchResults.length} 张相似度 >= ${SIMILARITY_THRESHOLD} 的图片。`;
                    displayedSearchResultsCount = 0;
                    imageGallery.innerHTML = ''; // Ensure gallery is clear before appending
                    loadMoreSearchResults(); // Load the first batch
                } else {
                    searchStatus.textContent = `未找到相似度 >= ${SIMILARITY_THRESHOLD} 的图片。`;
                    imageGallery.innerHTML = `<p>未找到与描述匹配且相似度足够高的图片。</p>`;
                    currentSearchResults = [];
                }
            }
        })
        .catch(error => {
            console.error('搜索错误:', error);
            searchStatus.textContent = '搜索过程中发生网络错误。';
            imageGallery.innerHTML = '<p>搜索过程中发生网络错误。</p>';
            currentSearchResults = [];
        })
        .finally(() => {
            loadingGallery.style.display = 'none';
            searchButton.disabled = false;
        });
    }

    function loadMoreSearchResults() {
        if (isLoadingMoreSearchResults) return;
        isLoadingMoreSearchResults = true;
        loadingGallery.style.display = 'block'; // Show loading indicator

        const nextBatch = currentSearchResults.slice(
            displayedSearchResultsCount,
            displayedSearchResultsCount + searchResultsBatchSize
        );

        if (nextBatch.length > 0) {
            displayImages(nextBatch, true, true); // Append = true
            displayedSearchResultsCount += nextBatch.length;
            noMoreResultsDiv.style.display = 'none';
        } else {
            if (displayedSearchResultsCount > 0 && currentSearchResults.length > 0 && displayedSearchResultsCount >= currentSearchResults.length) {
                 noMoreResultsDiv.style.display = 'block';
            } else if (displayedSearchResultsCount === 0 && currentSearchResults.length === 0) {
                // This case handled by performSearch's "not found" message
                noMoreResultsDiv.style.display = 'none';
            }
        }
        isLoadingMoreSearchResults = false;
        loadingGallery.style.display = 'none';
    }


    // Modified displayImages to support appending for infinite scroll
    function displayImages(images, isSearchResult = false, append = false) {
        if (!append) {
            imageGallery.innerHTML = ''; // Clear for new content (gallery or first batch of search)
        }

        if (!images || images.length === 0) {
            if (!append && isSearchResult) imageGallery.innerHTML = `<p>未找到相似度 >= ${SIMILARITY_THRESHOLD} 的图片。</p>`;
            else if (!append && !isSearchResult) imageGallery.innerHTML = '<p>图片库为空。</p>';
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
            imgElement.src = img.thumbnail_url || 'https://placehold.co/150x120/eee/ccc?text=NoThumb';
            imgElement.alt = img.filename;
            imgElement.onerror = () => { imgElement.src = 'https://placehold.co/150x120/eee/ccc?text=Error'; };

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

        modalImageElement.src = originalUrl || ''; // Handle if originalUrl is null/undefined
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
                    alert(`获取图片详情失败: ${data.error}`);
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
                alert('获取图片详情时发生网络错误。');
            });
    }

    closeModalButton.addEventListener('click', () => {
        modal.style.display = 'none';
        currentModalImageId = null;
    });

    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
            currentModalImageId = null;
        }
    });

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
                    // Update gallery item badge without full reload
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

    function updateGalleryPaginationControls(totalImages, currentPageNum, imagesPerPageNum, totalPagesCalculated) {
        galleryTotalPages = totalPagesCalculated > 0 ? totalPagesCalculated : 1;
        totalImagesCountSpan.textContent = totalImages; // This might be overridden by search status
        currentPageInfoSpan.textContent = `第 ${currentPageNum} / ${galleryTotalPages} 页`;
        prevPageButton.disabled = currentPageNum === 1;
        nextPageButton.disabled = currentPageNum === galleryTotalPages || totalImages === 0;
    }

    prevPageButton.addEventListener('click', () => {
        if (galleryCurrentPage > 1) {
            loadGalleryImages(galleryCurrentPage - 1);
        }
    });

    nextPageButton.addEventListener('click', () => {
        if (galleryCurrentPage < galleryTotalPages) {
            loadGalleryImages(galleryCurrentPage + 1);
        }
    });

    function switchToGalleryView() {
        currentSearchResults = []; // Clear search results
        displayedSearchResultsCount = 0;
        imageGallery.innerHTML = '';
        searchInput.value = ''; // Clear search input
        searchStatus.textContent = '';
        searchResultsTitleSpan.style.display = 'none';
        paginationControls.style.display = 'block'; // Show pagination for gallery
        noMoreResultsDiv.style.display = 'none';
        loadGalleryImages(1); // Load first page of gallery
    }


    function loadGalleryImages(page = 1) {
        galleryCurrentPage = page;
        imageGallery.innerHTML = ''; // Clear previous page's images
        loadingGallery.style.display = 'block';
        searchStatus.textContent = ''; // Clear any search status
        searchResultsTitleSpan.style.display = 'none';
        paginationControls.style.display = 'block';
        noMoreResultsDiv.style.display = 'none';


        fetch(`/images?page=${galleryCurrentPage}&limit=${galleryImagesPerPage}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    imageGallery.innerHTML = `<p>加载图片库失败: ${data.error}</p>`;
                    updateGalleryPaginationControls(0, 1, galleryImagesPerPage, 1);
                } else {
                    displayImages(data.images, false, false); // isSearchResult=false, append=false
                    updateGalleryPaginationControls(data.total_count, data.page, data.limit, data.total_pages);
                    totalImagesCountSpan.textContent = data.total_count; // Update total count for gallery
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

    // Infinite scroll for search results
    window.addEventListener('scroll', () => {
        // Check if we are in search mode and not already loading
        if (currentSearchResults.length > 0 && !isLoadingMoreSearchResults) {
            if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 300) { // Near bottom
                if (displayedSearchResultsCount < currentSearchResults.length) {
                    loadMoreSearchResults();
                } else if (displayedSearchResultsCount > 0) { // All loaded
                    noMoreResultsDiv.style.display = 'block';
                }
            }
        }
    });

    // Initial load: switch to gallery view
    switchToGalleryView();
});