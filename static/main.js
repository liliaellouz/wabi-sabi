$(document).ready(function(){
    $('#action_menu_btn').click(function(){
        $('.action_menu').toggle();
    });

    $('li#reset_convo').click(function() {
        window.location.replace('/reset');
        console.log('reset');
    });
});