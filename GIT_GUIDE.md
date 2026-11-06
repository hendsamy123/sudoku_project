# دليل Git للمشروع (انسخي الأوامر في PowerShell داخل فولدر المشروع)

## مرة واحدة لكل شخص
git --version
git config --global user.name  "اسمك بالإنجليزي"
git config --global user.email "نفس إيميل GitHub"

## الشخص الأول (ينشئ المشروع)
git init
git branch -M main
git add .
git commit -m "Initial commit: project skeleton"
# اعملي repo فاضي (Private) على GitHub ثم:
git remote add origin https://github.com/USERNAME/sudoku_project.git
git push -u origin main
# بعدين: GitHub > Settings > Collaborators > Add people

## باقي الفريق
git clone https://github.com/USERNAME/sudoku_project.git
cd sudoku_project

## الروتين اليومي
git pull
git status
git add <الملفات اللي اشتغلتي عليها>
git commit -m "رسالة قصيرة بالإنجليزي تقول عملتي إيه"
git push

## لو push اترفض
git pull --rebase
git push

## الـ git log للتسليم (انتبهي: > في Windows PowerShell بيطلّع UTF-16)
git log --stat --date=short | Out-File -Encoding utf8 git_log.txt
git shortlog -sn
git log --oneline --graph --decorate
