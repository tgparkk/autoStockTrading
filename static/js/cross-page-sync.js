/**
 * 페이지 간 동기화를 위한 스크립트
 * 설정 페이지와 대시보드 페이지 간 실시간 데이터 동기화를 지원합니다.
 */

// 다른 페이지에서 오는 메시지 수신 처리
window.addEventListener('message', function(event) {
    // 소스 검증이 필요하면 event.origin 확인
    try {
        if (event.data && event.data.type === 'REFRESH_DASHBOARD') {
            console.log('다른 페이지에서 갱신 요청 받음');
            
            // 대시보드 데이터 갱신 함수가 있는 경우
            if (typeof fetchDashboardData === 'function') {
                fetchDashboardData();
            }
            
            // 종목 목록 갱신 함수가 있는 경우
            if (typeof fetchStocksList === 'function') {
                fetchStocksList();
            }
            
            // 알림 표시 함수가 있는 경우
            if (typeof addAlert === 'function') {
                addAlert('info', '설정 페이지에서 종목 데이터가 갱신되었습니다.');
            }
        }
    } catch (e) {
        console.error('교차 페이지 메시지 처리 오류:', e);
    }
});

// 다른 페이지에 갱신 요청 전송
function notifyOtherPages(messageType = 'REFRESH_DASHBOARD') {
    try {
        // 모든 창(탭)에 메시지 브로드캐스트
        if (window.opener && !window.opener.closed) {
            // 부모 창이 있는 경우
            window.opener.postMessage({ type: messageType }, '*');
            console.log('부모 창에 갱신 요청 전송');
        }
        
        // 현재 창에서 연 다른 창들에 메시지 전송
        if (window.frames && window.frames.length > 0) {
            for (let i = 0; i < window.frames.length; i++) {
                try {
                    if (!window.frames[i].closed) {
                        window.frames[i].postMessage({ type: messageType }, '*');
                    }
                } catch (frameError) {
                    // 접근 제한된 프레임은 무시
                    console.log('프레임 접근 오류, 무시됨');
                }
            }
            console.log('자식 창에 갱신 요청 전송');
        }
    } catch (e) {
        console.error('다른 창에 메시지 전송 중 오류:', e);
    }
}

// 현재 페이지가 갱신될 때 다른 페이지에 알림
function notifyOnDataChange() {
    // 종목 목록이 업데이트되거나 설정이 변경될 때 호출
    notifyOtherPages('REFRESH_DASHBOARD');
}
