$(document).ready(function(){
    var from_quotes = false;
    $('#persona_sel').hide();

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
    });

    $('ul#change_persona li#from_quotes').click(function() {
        from_quotes = true
        $('.action_menu').toggle();
        $('#persona_sel').show();
        $('#current_persona').hide();
    });

    $('ul#change_persona li#from_wiki').click(function() {
        from_quotes = false
        $('.action_menu').toggle();
        $('#persona_sel').show();
        $('#current_persona').hide();
    });

    $('input#persona_in').on('input', function(e){
        val = $('input#persona_in').val()
        flag = false
        // check if list option
        options = $('#personas').children();
        for(var i = 0; i < options.length && !flag; i++) {
            // valid personna
            if(options[i].value === val) {
                flag = true
                // build persona
                if (from_quotes) {
                    $.post( 
                        '\change_persona_with_quotes',
                        { input: val },
                        function(data, status) {
                            location.reload();
                        }
                    );
                } else {
                    $.post( 
                        '\change_persona_with_backstory',
                        { input: val },
                        function(data, status) {
                            location.reload();
                        }
                    );
                }
            }
        }

        // if not list option
        if (!flag) {
            // get persona suggestions
            $.post( 
                '\get_personas',
                { input: val },
                function(data, status) {
                    data = JSON.parse(data);
                    // console.log("Personas received; " +data);
                    $('#personas').empty()
                    for (var i = 0; i < data.length; i++) {
                        $('#personas').append('<option value="'+data[i]+'">')
                    }
                }
            );
        }
    });

    $('div#send_btn').click(function(){
        input_txt = $('#input_str').val();
        $('#input_str').val('')
        push_str(input_txt, true);
        $.post( 
            '\get_answer',
            { input: input_txt },
            function(data, status) {
                push_str(data.replace(/["]+/g, ''), false);
            }
        );
        
    });
    $(document).keypress(function(e) {
        // if user pressed enter
        if(e.which == 13) {
            e.preventDefault();
            input_txt = $('#input_str').val();
            $('#input_str').val('')
            push_str(input_txt, true);
            $.post( 
                '\get_answer',
                { input: input_txt },
                function(data, status) {
                    push_str(data.replace(/["]+/g, ''), false);
                }
            );
        }
    });
});