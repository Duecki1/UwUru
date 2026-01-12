<div class='content-wrapper' id='post'>
    <h1>Post #<%- ctx.post.id %></h1>
    <nav class='buttons'><!--
        --><ul><!--
            --><li><a href='<%- ctx.formatClientLink('post', ctx.post.id) %>'><i class='fa fa-reply'></i> Main view</a></li><!--
            --><% if (ctx.canMerge) { %><!--
                --><li data-name='merge'><a href='<%- ctx.formatClientLink('post', ctx.post.id, 'merge') %>'>Merge with&hellip;</a></li><!--
            --><% } %><!--
            --><% if (ctx.canDelete) { %><!--
                --><li><a href class='delete-post'><i class='fa fa-trash'></i> Delete</a></li><!--
            --><% } %><!--
        --></ul><!--
    --></nav>
    <div class='post-content-holder'></div>
</div>
