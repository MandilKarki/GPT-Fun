rm ../gpt_service.zip
zip ../gpt_service.zip  -r * .[^.]* -x "virt/*" ".git/*" "my_virtual_environment/*" ".ebignore" \
 ".gitignore" ".vscode/*" "files/*" ".idea/*" "data/*" "*__pycache__*"

