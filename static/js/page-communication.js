/**
 * 페이지 통신 유틸리티
 * 설정 페이지와 대시보드 페이지 간의 통신을 지원합니다.
 */

/**
 * 설정 페이지에서 종목이 업데이트될 때 대시보드 페이지도 갱신하기 위한 함수
 * @param {string} message 대시보드로 전송할 메시지
 */
function notifyDashboard(message) {
    try {
        // 새 창에서 열린 대시보드에 메시지 전송
        if (window.opener && !window.opener.closed) {
            window.opener.postMessage({ type: 'REFRESH_DASHBOARD', message: message }, '*');
            console.log('부모 창으로 데이터 업데이트 알림 전송됨');
        }
        
        // 같은 도메인의 다른 창 찾기 시도
        for (let i = 0; i < window.top.frames.length; i++) {
            try {
                if (!window.top.frames[i].closed) {
                    window.top.frames[i].postMessage({ type: 'REFRESH_DASHBOARD', message: message }, '*');
                }
            } catch (e) {
                // 동일 출처 정책으로 인한 오류는 무시
                console.debug('프레임 접근 제한됨');
            }
        }
    } catch (e) {
        console.error('대시보드 알림 실패:', e);
    }
}

/**
 * 대시보드 페이지에서 메시지 받아 처리하는 함수
 */
function setupDashboardListener() {
    window.addEventListener('message', function(event) {
        // 필요한 경우 origin 검증
        try {
            if (event.data && event.data.type === 'REFRESH_DASHBOARD') {
                console.log('설정 페이지에서 업데이트 알림 받음:', event.data.message || '');
                
                // 대시보드 데이터 갱신
                if (typeof fetchDashboardData === 'function') {
                    fetchDashboardData();
                    console.log('대시보드 데이터 갱신됨');
                }
                
                // 종목 목록 갱신
                if (typeof fetchStocksList === 'function') {
                    fetchStocksList();
                    console.log('종목 목록 갱신됨');
                }
                
                // 알림 표시 (있는 경우)
                if (typeof addAlert === 'function') {
                    addAlert('info', event.data.message || '설정 페이지에서 데이터가 업데이트되었습니다.');
                }
            }
        } catch (e) {
            console.error('메시지 처리 오류:', e);
        }
    });
    
    console.log('대시보드 메시지 리스너 설정 완료');
}

// 페이지 로드 시 자동으로 리스너 설정 (대시보드에서 호출됨)
document.addEventListener('DOMContentLoaded', function() {
    // 현재 페이지가 대시보드인지 확인 (URL로 구분)
    if (window.location.href.includes('/dashboard')) {
        setupDashboardListener();
    }
});