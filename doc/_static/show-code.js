// Stolen from statsmodels and fixed up some

function cleanQuotes(text){
    /// Replace all single and double quotes with one escaped backslash
    return text.replace(/(["'])/g, "\\\1");
}

function scrapeText(codebox){
    /// Returns input lines cleaned of prompt1 and prompt2
    var lines = codebox.split('\n');
    var newlines = new Array();
    $.each(lines, function() {
        if (this.match(/^In \[\d+]: /)){
            newlines.push(cleanQuotes(this.replace(/^(\s)*In \[\d+]: /,"")));
        }
        else if (this.match(/^(\s)*\.+:/)){
            newlines.push(cleanQuotes(this.replace(/^(\s)*\.+: /,"")));
        }

    }
            );
    return newlines.join('\\n');
}

$(document).ready(            
        function() {
    // grab all code boxes
    var ipythoncode = $(".highlight-ipython");
    $.each(ipythoncode, function() {
        //var codebox = cleanUpText($(this).text());
        var codebox = scrapeText($(this).text());
        // give them a facebox pop-up with plain text code   
        $(this).append('<span style="text-align:right; display:block; margin-top:-10px; margin-left:10px; font-size:60%"><a href="javascript: jQuery.facebox(\'<textarea cols=80 rows=10 readonly style=margin:5px onmouseover=javascript:this.select();>'+codebox+'</textarea>\');">View Code</a></span>');
        $(this,"textarea").select();
    });
});
