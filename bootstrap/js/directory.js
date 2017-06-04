$( document ).ready(function() {
    var out="";

    $(".post-direct").each(function(i, obj) {
        out+=i;
    });

    $("#loadfiles").html(out);
});

