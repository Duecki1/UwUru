<div class='file-dropper-holder'>
    <input type='file' id='<%- ctx.id %>'/>
    <label class='file-dropper' for='<%- ctx.id %>' role='button'>
        <% if (ctx.allowMultiple) { %>
            Drop files here!
        <% } else { %>
            Drop file here!
        <% } %>
        <br/>
        Or just click on this box.
        <% if (ctx.extraText) { %>
            <br/>
            <small><%= ctx.extraText %></small>
        <% } %>
    </label>
    <% if (ctx.allowUrls) { %>
        <div class='url-holder'>
            <input type='text' name='url' placeholder='<%- ctx.urlPlaceholder %>' title='Tip: X/Twitter, Instagram, and similar links can be fetched when the server downloader is enabled.'/>
            <% if (ctx.lock) { %>
                <button title='Tip: X/Twitter, Instagram, and similar links can be fetched when the server downloader is enabled.'>Confirm</button>
            <% } else { %>
                <button title='Tip: X/Twitter, Instagram, and similar links can be fetched when the server downloader is enabled.'>Add URL</button>
            <% } %>
        </div>
    <% } %>
</div>
