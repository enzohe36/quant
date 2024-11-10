git stash
git pull
git stash pop
git add -A
read -p "Commit note: " message
git commit -m "$message$"
git push