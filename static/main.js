$(document).ready(function(){

    function push_str(text, is_input) {
        if (is_input) {
            // add as user text
            $('#conversation').append(`
            <div class="card-body msg_card_body" id="conversation">
            <div class="d-flex justify-content-end mb-4">
                <div class="msg_cotainer_send">
                    `+text+`
                    <!-- <span class="msg_time_send">8:55 AM, Today</span> -->
                </div>
                <div class="img_cont_msg">
            <img src="/static/user.png" class="rounded-circle user_img_msg">
                </div>
            </div>
            `);
        } else {
            // add as ai text
            $('#conversation').append(`
            <div class="d-flex justify-content-start mb-4">
            <div class="img_cont_msg">
                <img src="/static/wabisabi.png" class="rounded-circle user_img_msg">
            </div>
            <div class="msg_cotainer">
                `+text+`
                <!-- <span class="msg_time">8:40 AM, Today</span> -->
            </div>
        </div>
            `);
        }
    }

    $('#action_menu_btn').click(function(){
        $('.action_menu').toggle();
    });

    $('li#reset_convo').click(function() {
        window.location.replace('/reset');
        console.log('reset');
    });

    $('div#send_btn').click(function(){
        input_txt = $('#input_str').val();
        $('#input_str').val('')
        push_str(input_txt, true);
        $.post( '\answer', // 'https://ptsv2.com/t/70v19-1573937191/post',
                { 
                    input: input_txt
                },
                function(data, status) {
                    push_str(data.replace(/["]+/g, ''), false);
                })
        
    });
});