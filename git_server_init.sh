#/bin/sh

if [ $# -ne 1 ]
then
	echo "USAGE: $0 <Porject Name>"
	exit
fi

cd $1

git init
git add .
git commit -m "initial commit"
cd ..

git clone --bare $1 $1.git && scp -r $1.git uwsn:~/gits/.&& rm -rf $1.git

cd $1
git remote add uwsn uwsn:~/gits/$1.git && git push -u uwsn master
# setting tracking branch
#git push -u uwsn master
