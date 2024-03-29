function [] = discordLog(message)
    webhook = 'https://discord.com/api/webhooks/1217693168234008606/VxbqGnx5jeo7WVO-futht66b3WNHYV25ifJNI8Slkky8ZmPvIVqzLQCg2VKCTzH0cFi0';

    % Set up the options for the web request
    options = weboptions('MediaType','application/json');

    try
        % Try to send the POST request
        webwrite(webhook, struct('content', message), options);
    catch
        % If an error occurs, this block is executed
        warning('Failed to log message to Discord.');
    end
end