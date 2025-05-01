// 소켓 연결
const socket = io();

// 진행 상태 타이머 ID를 저장할 전역 변수
let progressTimers = [];

// 진행 상태 표시 함수
function updateProgressStatus(message, progress) {
    const statusDiv = document.getElementById('updateStatus');
    statusDiv.innerHTML = `
        <div class="alert alert-info">
        ${message}
        <div class="progress mt-2">
            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                role="progressbar" style="width: ${progress}%" 
                aria-valuenow="${progress}" aria-valuemin="0" aria-valuemax="100">
            ${progress}%
            </div>
        </div>
        </div>
    `;
}

// 에러 메시지 표시 - 자동으로 사라지지 않음
function showErrorMessage(message, details = null, strategyType = null) {
    // 먼저 진행중인 모든 타이머 취소
    clearAllProgressTimers();
    
    console.error('오류:', message);
    
    // 오류 상세 정보 표시 개선
    let errorMsg = `<div class="alert alert-danger alert-dismissible fade show">
                   <h5><i class="bi bi-exclamation-triangle"></i> 종목 재선정 실패</h5>
                   <p>${message}</p>`;

    // 상세 오류 정보가 있는 경우 표시
    if (details) {
        errorMsg += `<div class="small text-muted mt-2">
                        <strong>오류 상세:</strong> ${details}
                      </div>`;
    }
    
    // 전략 유형 정보 표시
    if (strategyType) {
        errorMsg += `<div class="small text-muted">
                        <strong>전략 유형:</strong> ${strategyType}
                      </div>`;
    }
    
    // 재시도 버튼 추가
    errorMsg += `<div class="mt-2">
                   <button class="btn btn-sm btn-outline-danger retry-update-btn me-2">
                     <i class="bi bi-arrow-repeat"></i> 재시도
                   </button>
                   <button class="btn btn-sm btn-outline-secondary close-error-btn">
                     <i class="bi bi-x-circle"></i> 닫기
                   </button>
                </div>
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>`;
    
    document.getElementById('updateStatus').innerHTML = errorMsg;
    
    // 재시도 버튼 이벤트 리스너 추가
    document.querySelector('.retry-update-btn').addEventListener('click', function() {
        document.getElementById('updateStocksBtn').click();
    });
    
    // 닫기 버튼 이벤트 리스너 추가
    document.querySelector('.close-error-btn').addEventListener('click', function() {
        document.getElementById('updateStatus').innerHTML = '';
    });
}

// 모든 진행 타이머 취소 함수
function clearAllProgressTimers() {
    if(progressTimers && progressTimers.length > 0) {
        progressTimers.forEach(timerId => clearTimeout(timerId));
        progressTimers = [];
    }
}

// 성공 메시지 표시 - 5초 후 자동으로 사라짐
function showSuccessMessage(message) {
    const statusElem = document.getElementById('updateStatus');
    statusElem.innerHTML = `<div class="alert alert-success">${message}</div>`;
    
    // 5초 후 성공 메시지만 사라짐
    setTimeout(() => {
        if (statusElem.querySelector('.alert-success')) {
            statusElem.innerHTML = '';
        }
    }, 5000);
}

// 에러 처리 강화 함수
function handleApiError(error, operation) {
    // 먼저 진행중인 모든 타이머 취소
    clearAllProgressTimers();
    
    console.error('Error:', error);
    
    let errorMessage = '알 수 없는 오류가 발생했습니다.';
    
    // 네트워크 오류 처리
    if (error.name === 'TypeError' && error.message.includes('network')) {
        errorMessage = '네트워크 연결을 확인해주세요.';
    } 
    // 시간 초과 오류
    else if (error.name === 'TimeoutError') {
        errorMessage = '요청 시간이 초과되었습니다.';
    } 
    // 기타 오류
    else {
        errorMessage = `오류: ${error.message || '알 수 없는 오류'}`;
    }
    
    // 개선된 오류 메시지 UI
    const errorHTML = `
    <div class="alert alert-danger alert-dismissible fade show">
        <h5><i class="bi bi-exclamation-triangle"></i> ${operation} 중 오류 발생</h5>
        <p>${errorMessage}</p>
        <div class="mt-2">
            <button class="btn btn-sm btn-outline-danger retry-btn me-2">
                <i class="bi bi-arrow-repeat"></i> 재시도
            </button>
            <button class="btn btn-sm btn-outline-secondary close-btn">
                <i class="bi bi-x-circle"></i> 닫기
            </button>
        </div>
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    </div>`;
    
    const updateStatus = document.getElementById('updateStatus');
    updateStatus.innerHTML = errorHTML;
    
    // 재시도 버튼 이벤트 리스너 추가
    updateStatus.querySelector('.retry-btn').addEventListener('click', function() {
        // 작업 재시도 로직
        if (operation === '종목 목록 조회') {
            document.getElementById('refreshStocksBtn').click();
        } else if (operation === '종목 재선정 및 갱신') {
            document.getElementById('updateStocksBtn').click();
        }
    });
    
    // 닫기 버튼 이벤트 리스너 추가
    updateStatus.querySelector('.close-btn').addEventListener('click', function() {
        updateStatus.innerHTML = '';
    });
}

// 시스템 상태에 따라 버튼 상태 업데이트 함수
function updateButtonState(status) {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    if (status === 'running') {
        startBtn.disabled = true;
        startBtn.classList.remove('btn-success');
        startBtn.classList.add('btn-secondary');
        stopBtn.disabled = false;
        stopBtn.classList.remove('btn-secondary');
        stopBtn.classList.add('btn-danger');
    } else if (status === 'waiting') {
        startBtn.disabled = true;
        startBtn.classList.remove('btn-success');
        startBtn.classList.add('btn-secondary');
        stopBtn.disabled = false;
        stopBtn.classList.remove('btn-secondary');
        stopBtn.classList.add('btn-danger');
    } else { // stopped
        startBtn.disabled = false;
        startBtn.classList.remove('btn-secondary');
        startBtn.classList.add('btn-success');
        stopBtn.disabled = true;
        stopBtn.classList.remove('btn-danger');
        stopBtn.classList.add('btn-secondary');
    }
}

// 알림 메시지 표시 함수
function showNotification(type, message, duration = 5000) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // 시스템 정보 카드 아래에 삽입
    const systemCard = document.querySelector('.card-header.bg-info').parentNode;
    systemCard.appendChild(alertDiv);
    
    // 지정된 시간 후 자동 제거
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 300);
    }, duration);
}

// 시스템 상태 업데이트
socket.on('system_status', function(data) {
    let statusText = '';
    
    if (data.status === 'running') {
        statusText = '<i class="bi bi-play-circle-fill text-success"></i> 실행 중';
    } else if (data.status === 'waiting') {
        statusText = '<i class="bi bi-pause-circle-fill text-warning"></i> 대기 중';
    } else {
        statusText = '<i class="bi bi-stop-circle-fill text-danger"></i> 중지됨';
    }
    
    document.getElementById('systemStatus').innerHTML = statusText;
    document.getElementById('lastUpdate').textContent = data.last_update;
    
    // 시장 국면 표시
    if (data.market_regime) {
        let regimeText = '';
        if (data.market_regime === 'bullish') {
            regimeText = '<span class="text-success">강세장</span>';
        } else if (data.market_regime === 'bearish') {
            regimeText = '<span class="text-danger">약세장</span>';
        } else {
            regimeText = '<span class="text-warning">중립</span>';
        }
        document.getElementById('marketRegime').innerHTML = regimeText;
    }
    
    // 다음 업데이트 예정 시간
    if (data.next_update) {
        document.getElementById('nextUpdate').textContent = data.next_update;
    }
    
    // 버튼 상태 업데이트
    updateButtonState(data.status);
    
    // 시스템 상태 변화에 따라 알림 표시 (새로고침 없이 상태 변화 표시)
    if (data.status === 'running' && document.getElementById('startBtn').disabled === false) {
        showNotification('success', '<i class="bi bi-info-circle"></i> 자동매매가 시작되었습니다.');
        // 버튼 텍스트 복원
        document.getElementById('startBtn').innerHTML = '자동매매 시작';
    } else if (data.status === 'stopped' && document.getElementById('stopBtn').disabled === false) {
        showNotification('warning', '<i class="bi bi-info-circle"></i> 자동매매가 중지되었습니다.');
        // 버튼 텍스트 복원
        document.getElementById('stopBtn').innerHTML = '자동매매 중지';
    }
});

// 종목 정보 업데이트
socket.on('selected_stocks_update', function(data) {
    const tbody = document.getElementById('selectedStocksTableBody');
    tbody.innerHTML = '';
    
    // 업데이트 시간 표시 추가
    const updateTimeDisplay = document.createElement('div');
    updateTimeDisplay.textContent = `마지막 업데이트: ${new Date().toLocaleTimeString()}`;
    document.getElementById('updateTimeContainer').innerHTML = '';
    document.getElementById('updateTimeContainer').appendChild(updateTimeDisplay);
    
    // 애니메이션 효과로 업데이트 강조
    document.getElementById('selectedStocksTable').classList.add('table-update-flash');
    setTimeout(() => {
        document.getElementById('selectedStocksTable').classList.remove('table-update-flash');
    }, 1000);
    
    data.forEach(stock => {
        const row = document.createElement('tr');
        
        // 수익률 데이터가 있으면 색상 적용
        let profitClass = '';
        let profitDisplay = '';
        if (stock.profit_ratio) {
            profitClass = stock.profit_ratio > 0 ? 'text-success' : 'text-danger';
            profitDisplay = `<span class="${profitClass}">${(stock.profit_ratio * 100).toFixed(2)}%</span>`;
        }
        
        // 선정 점수 포맷팅
        let scoreDisplay = '-';
        if (stock.score !== null) {
            // 소수점 둘째 자리까지 표시
            scoreDisplay = stock.score.toFixed(2);
        }
        
        row.innerHTML = `
            <td>${stock.code}</td>
            <td>${stock.name || '-'}</td>
            <td>${stock.selected_date || '-'}</td>
            <td>${scoreDisplay}</td>
        `;
        tbody.appendChild(row);
    });
    
    document.getElementById('stockCount').textContent = data.length;
});

// 전략 목록 로드 함수
function loadStrategies() {
    fetch('/api/strategy/list')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const strategiesSelect = document.getElementById('strategySelect');
                strategiesSelect.innerHTML = '<option value="">전략을 선택하세요</option>';
                
                // 현재 전략 표시
                document.getElementById('currentStrategyName').textContent = 
                    data.current_strategy.name || '알 수 없음';
                
                // 사용 가능한 전략 목록 채우기
                data.available_strategies.forEach(strategy => {
                    const option = document.createElement('option');
                    option.value = strategy.name;
                    option.textContent = strategy.name;
                    option.dataset.description = strategy.description;
                    option.dataset.features = strategy.features ? strategy.features.join(', ') : '';
                    option.dataset.isCurrent = strategy.is_current;
                    
                    // 현재 전략인 경우 선택 표시
                    if (strategy.is_current) {
                        option.textContent = `${strategy.name} (현재 사용 중)`;
                        option.disabled = true;
                    }
                    
                    strategiesSelect.appendChild(option);
                });
                
                // 전략 변경 버튼 상태 업데이트
                updateChangeButtonState();
            } else {
                showErrorMessage('전략 목록을 불러오는 데 실패했습니다: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error loading strategies:', error);
            showErrorMessage('전략 목록을 불러오는 중 오류가 발생했습니다.');
        });
}

// 전략 선택 변경 시 설명 업데이트
function updateStrategyDescription() {
    const select = document.getElementById('strategySelect');
    const selectedOption = select.options[select.selectedIndex];
    const description = selectedOption.dataset.description || '설명이 없습니다.';
    const features = selectedOption.dataset.features || '';
    
    let descriptionHTML = `<p>${description}</p>`;
    
    if (features) {
        descriptionHTML += `<p><strong>특징:</strong> ${features}</p>`;
    }
    
    document.getElementById('strategyDescription').innerHTML = descriptionHTML;
    
    // 변경 버튼 상태 업데이트
    updateChangeButtonState();
}

// 전략 변경 버튼 상태 업데이트
function updateChangeButtonState() {
    const select = document.getElementById('strategySelect');
    const changeBtn = document.getElementById('changeStrategyBtn');
    
    if (select.value === '') {
        changeBtn.disabled = true;
        changeBtn.classList.add('btn-secondary');
        changeBtn.classList.remove('btn-primary');
    } else {
        const selectedOption = select.options[select.selectedIndex];
        if (selectedOption.dataset.isCurrent === 'true') {
            changeBtn.disabled = true;
            changeBtn.classList.add('btn-secondary');
            changeBtn.classList.remove('btn-primary');
        } else {
            changeBtn.disabled = false;
            changeBtn.classList.add('btn-primary');
            changeBtn.classList.remove('btn-secondary');
        }
    }
}

// 전략 변경 함수
function changeStrategy() {
    const select = document.getElementById('strategySelect');
    const selectedStrategy = select.value;
    
    if (!selectedStrategy) {
        showErrorMessage('변경할 전략을 선택해주세요.');
        return;
    }
    
    // 사용자 확인
    if (!confirm(`'${selectedStrategy}' 전략으로 변경하시겠습니까?\n\n주의: 전략 변경은 현재 실행 중인 거래에 영향을 줄 수 있습니다.`)) {
        return;
    }
    
    // 상태 표시
    const changeBtn = document.getElementById('changeStrategyBtn');
    changeBtn.disabled = true;
    changeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 변경 중...';
    
    document.getElementById('strategyChangeStatus').innerHTML = 
        '<div class="alert alert-info">전략 변경 중입니다...</div>';
    
    // API 호출
    fetch('/api/strategy/change', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            strategy_name: selectedStrategy
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('strategyChangeStatus').innerHTML = 
                `<div class="alert alert-success">${data.message}</div>`;
                
            // 전략 목록 다시 로드
            loadStrategies();
            
            // 시스템 상태 확인
            fetch('/api/system/status')
                .then(response => response.json())
                .then(statusData => {
                    // 상태 정보 업데이트
                    if (statusData.status) {
                        updateButtonState(statusData.status);
                        
                        // 시스템 상태 표시 업데이트
                        let statusText = '';
                        if (statusData.status === 'running') {
                            statusText = '<i class="bi bi-play-circle-fill text-success"></i> 실행 중';
                        } else if (statusData.status === 'waiting') {
                            statusText = '<i class="bi bi-pause-circle-fill text-warning"></i> 대기 중';
                        } else {
                            statusText = '<i class="bi bi-stop-circle-fill text-danger"></i> 중지됨';
                        }
                        document.getElementById('systemStatus').innerHTML = statusText;
                    }
                })
                .catch(error => console.error('상태 확인 오류:', error));
                
        } else {
            document.getElementById('strategyChangeStatus').innerHTML = 
                `<div class="alert alert-danger">${data.message}</div>`;
        }
    })
    .catch(error => {
        console.error('Error changing strategy:', error);
        document.getElementById('strategyChangeStatus').innerHTML = 
            '<div class="alert alert-danger">전략 변경 중 오류가 발생했습니다.</div>';
    })
    .finally(() => {
        // 버튼 상태 복원
        changeBtn.disabled = false;
        changeBtn.innerHTML = '전략 변경하기';
        
        // 5초 후 성공/실패 메시지 제거
        setTimeout(() => {
            const statusElem = document.getElementById('strategyChangeStatus');
            if (statusElem.querySelector('.alert')) {
                statusElem.innerHTML = '';
            }
        }, 5000);
    });
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    // 시스템 상태 확인 및 버튼 상태 초기화
    fetch('/api/system/status')
        .then(response => response.json())
        .then(data => {
            updateButtonState(data.status);
        })
        .catch(error => console.error('상태 확인 오류:', error));
        
    // 종목 목록 조회 버튼
    document.getElementById('refreshStocksBtn').addEventListener('click', function() {
        // 버튼에 로딩 상태 표시
        this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 불러오는 중...';
        this.disabled = true;
        
        fetch('/api/stocks/list')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showSuccessMessage(`종목 목록이 새로고침되었습니다. (${data.stocks.length}개 종목)`);
                
                // 업데이트 시간 표시
                const updateTimeDisplay = document.createElement('div');
                updateTimeDisplay.textContent = `마지막 업데이트: ${new Date().toLocaleTimeString()}`;
                document.getElementById('updateTimeContainer').innerHTML = '';
                document.getElementById('updateTimeContainer').appendChild(updateTimeDisplay);
                
                // 애니메이션 효과로 업데이트 강조
                document.getElementById('selectedStocksTable').classList.add('table-update-flash');
                setTimeout(() => {
                document.getElementById('selectedStocksTable').classList.remove('table-update-flash');
                }, 1000);
                    
                    // 성공 알림 표시
                    showSuccessMessage(`종목 목록 갱신 완료 (${data.stocks.length}개 종목)`);
                    
                    // 대시보드 페이지가 열려 있다면 갱신하도록 메시지 전송
                    notifyOtherPages('REFRESH_DASHBOARD');
            } else {
                showErrorMessage(data.message);
            }
        })
        .catch(error => {
            handleApiError(error, '종목 목록 조회');
        })
        .finally(() => {
            // 버튼 원래 상태로 복구
            this.innerHTML = '종목 목록 조회';
            this.disabled = false;
        });
    });
    
    // 종목 재선정 및 갱신 버튼
    document.getElementById('updateStocksBtn').addEventListener('click', function() {
        if (confirm('종목을 수동으로 갱신하시겠습니까? 시스템이 현재 기준으로 종목을 재선정합니다.')) {
            // 버튼에 로딩 상태 표시
            this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 갱신 중...';
            this.disabled = true;
            
            // 진행중인 모든 타이머 취소
            clearAllProgressTimers();
            
            // 진행 상태 표시 시작
            updateProgressStatus('종목 선정 준비 중...', 10);
            
            // 진행 상태 시뮬레이션 (실제로는 서버에서 진행 상태를 받아오는 것이 좋음)
            progressTimers.push(setTimeout(() => updateProgressStatus('후보 종목 분석 중...', 30), 1000));
            progressTimers.push(setTimeout(() => updateProgressStatus('최종 종목 선정 중...', 60), 2000));
            
            fetch('/api/stocks/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                // 진행중인 모든 타이머 취소
                clearAllProgressTimers();
                
                if (data.success) {
                    updateProgressStatus('종목 갱신 완료!', 100);
                    setTimeout(() => {
                        showSuccessMessage(data.message);
                        
                        // 다른 페이지에 업데이트 알림
                        notifyOtherPages('REFRESH_DASHBOARD');
                    }, 500);
                    
                    // 업데이트 시간 표시
                    const updateTimeDisplay = document.createElement('div');
                    updateTimeDisplay.textContent = `마지막 갱신: ${new Date().toLocaleTimeString()}`;
                    document.getElementById('updateTimeContainer').innerHTML = '';
                    document.getElementById('updateTimeContainer').appendChild(updateTimeDisplay);
                } else {
                    // 오류 발생 시 에러 메시지 표시
                    const details = data.details && data.details.error ? data.details.error : null;
                    const strategyType = data.details && data.details.strategy_type ? data.details.strategy_type : null;
                    showErrorMessage(data.message, details, strategyType);
                }
            })
            .catch(error => {
                // 진행중인 모든 타이머 취소
                clearAllProgressTimers();
                
                handleApiError(error, '종목 재선정 및 갱신');
            })
            .finally(() => {
                // 버튼 원래 상태로 복구
                this.innerHTML = '종목 재선정 및 갱신';
                this.disabled = false;
            });
        }
    });
    
    // 전략 설정 폼 제출
    document.getElementById('strategyForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        // 폼 데이터 수집
        const formData = {
            strategy: {
                ma_short: parseInt(document.getElementById('ma_short').value),
                ma_long: parseInt(document.getElementById('ma_long').value),
                rsi_period: parseInt(document.getElementById('rsi_period').value),
                rsi_oversold: parseInt(document.getElementById('rsi_oversold').value),
                rsi_overbought: parseInt(document.getElementById('rsi_overbought').value),
                bb_period: parseInt(document.getElementById('bb_period').value),
                bb_std: parseFloat(document.getElementById('bb_std').value),
                stop_loss: parseFloat(document.getElementById('stop_loss').value) / 100,
                take_profit: parseFloat(document.getElementById('take_profit').value) / 100,
                max_position: parseInt(document.getElementById('max_position').value),
                position_size: parseFloat(document.getElementById('position_size').value) / 100
            },
            interval: parseInt(document.getElementById('interval').value)
        };
        
        // API 호출
        fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert(data.message);
                // 다른 페이지에 설정 변경 알림
                notifyOtherPages('REFRESH_DASHBOARD');
            } else {
                alert('오류: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('설정 저장 중 오류가 발생했습니다.');
        });
    });

    // 시작 버튼 클릭 이벤트
    document.getElementById('startBtn').addEventListener('click', function() {
        // 버튼 로딩 상태 표시
        this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 시작 중...';
        this.disabled = true;
        
        fetch('/api/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // 성공 알림 표시
                showNotification('success', '<i class="bi bi-check-circle"></i> ' + data.message);
                
                // 서버에서 시스템 상태 재확인
                setTimeout(function() {
                    fetch('/api/system/status')
                        .then(response => response.json())
                        .then(statusData => {
                            // 시스템 상태 업데이트
                            let statusText = '';
                            if (statusData.status === 'running') {
                                statusText = '<i class="bi bi-play-circle-fill text-success"></i> 실행 중';
                            } else if (statusData.status === 'waiting') {
                                statusText = '<i class="bi bi-pause-circle-fill text-warning"></i> 대기 중';
                            } else {
                                statusText = '<i class="bi bi-stop-circle-fill text-danger"></i> 중지됨';
                            }
                            document.getElementById('systemStatus').innerHTML = statusText;
                            document.getElementById('lastUpdate').textContent = statusData.last_update;
                            
                            // 버튼 상태 업데이트
                            updateButtonState(statusData.status);
                        })
                        .catch(error => console.error('상태 확인 오류:', error));
                }, 500); // 0.5초 후 상태 확인
                
                // 버튼 텍스트 복원
                this.innerHTML = '자동매매 시작';
            } else {
                // 오류 알림 표시
                showNotification('danger', '<i class="bi bi-exclamation-triangle"></i> 오류: ' + data.message);
                // 버튼 원래 상태로 복구
                this.innerHTML = '자동매매 시작';
                this.disabled = false;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            // 오류 알림 표시
            showNotification('danger', '<i class="bi bi-exclamation-triangle"></i> 요청 처리 중 오류가 발생했습니다. 다시 시도해주세요.');
            // 버튼 원래 상태로 복구
            this.innerHTML = '자동매매 시작';
            this.disabled = false;
        });
    });

    // 중지 버튼 클릭 이벤트
    document.getElementById('stopBtn').addEventListener('click', function() {
        // 확인 대화상자
        if (confirm('자동매매를 중지하시겠습니까?')) {
            // 버튼 로딩 상태 표시
            this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 중지 중...';
            this.disabled = true;
            
            fetch('/api/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // 성공 알림 표시
                    showNotification('warning', '<i class="bi bi-check-circle"></i> ' + data.message);
                    
                    // 서버에서 시스템 상태 재확인
                    setTimeout(function() {
                        fetch('/api/system/status')
                            .then(response => response.json())
                            .then(statusData => {
                                // 시스템 상태 업데이트
                                let statusText = '';
                                if (statusData.status === 'running') {
                                    statusText = '<i class="bi bi-play-circle-fill text-success"></i> 실행 중';
                                } else if (statusData.status === 'waiting') {
                                    statusText = '<i class="bi bi-pause-circle-fill text-warning"></i> 대기 중';
                                } else {
                                    statusText = '<i class="bi bi-stop-circle-fill text-danger"></i> 중지됨';
                                }
                                document.getElementById('systemStatus').innerHTML = statusText;
                                document.getElementById('lastUpdate').textContent = statusData.last_update;
                                
                                // 버튼 상태 업데이트
                                updateButtonState(statusData.status);
                            })
                            .catch(error => console.error('상태 확인 오류:', error));
                    }, 500); // 0.5초 후 상태 확인
                    
                    // 버튼 텍스트 복원
                    this.innerHTML = '자동매매 중지';
                } else {
                    // 오류 알림 표시
                    showNotification('danger', '<i class="bi bi-exclamation-triangle"></i> 오류: ' + data.message);
                    // 버튼 원래 상태로 복구
                    this.innerHTML = '자동매매 중지';
                    this.disabled = false;
                }
            })
            .catch(error => {
                console.error('Error:', error);
                // 오류 알림 표시
                showNotification('danger', '<i class="bi bi-exclamation-triangle"></i> 요청 처리 중 오류가 발생했습니다. 다시 시도해주세요.');
                // 버튼 원래 상태로 복구
                this.innerHTML = '자동매매 중지';
                this.disabled = false;
            });
        }
    });
    
    // 전략 목록 초기 로드
    loadStrategies();
    
    // 전략 선택 변경 이벤트 리스너 추가
    document.getElementById('strategySelect').addEventListener('change', updateStrategyDescription);
    
    // 전략 변경 버튼 이벤트 리스너 추가
    document.getElementById('changeStrategyBtn').addEventListener('click', changeStrategy);
    
    // 자동 새로고침 기능
    let autoRefreshInterval;
    
    document.getElementById('autoRefreshCheck').addEventListener('change', function() {
        if (this.checked) {
            document.getElementById('updateStatus').innerHTML = 
                '<div class="alert alert-info">자동 새로고침이 활성화되었습니다 (30초 간격)</div>';
            
            // 30초마다 새로고침
            autoRefreshInterval = setInterval(() => {
                document.getElementById('refreshStocksBtn').click();
            }, 30000);
            
            // 3초 후 메시지 제거
            setTimeout(() => {
                const statusElem = document.getElementById('updateStatus');
                if (statusElem.querySelector('.alert-info')) {
                    statusElem.innerHTML = '';
                }
            }, 3000);
        } else {
            clearInterval(autoRefreshInterval);
            document.getElementById('updateStatus').innerHTML = 
                '<div class="alert alert-warning">자동 새로고침이 비활성화되었습니다</div>';
            
            // 3초 후 메시지 제거
            setTimeout(() => {
                const statusElem = document.getElementById('updateStatus');
                if (statusElem.querySelector('.alert-warning')) {
                    statusElem.innerHTML = '';
                }
            }, 3000);
        }
    });
});
