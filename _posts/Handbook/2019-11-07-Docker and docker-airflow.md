![title image](https://images.pexels.com/photos/67636/rose-blue-flower-rose-blooms-67636.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260)

### Background

A handbook for docker and puckle/docker-airflow

### Commands

| Goal                                             | Command                                                      | Note                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| stop and remove all container                    | `docker stop $(docker ps -a -q)`<br>`docker rm $(docker ps -a -q)` | this is in linux, if in windows, you need ``for /F "usebackq delims=" %A in (`docker ps -a -q`) do docker rm %A`` |
| to inspect file system of a container            | `docker container ls`<br>`docker exec -it 'names' bash`      | use 1st command to get the 'names' of the container, then run 2nd command; then you can use ls, cd or any command  you like |
| run container docker-airflow with local executor | `docker-compose -f docker-compose-LocalExecutor.yml up -d`   | first need go into the folder of **.yml                      |
| list all docker containers                       | `docker container ls`                                        |                                                              |
|                                                  |                                                              |                                                              |
|                                                  |                                                              |                                                              |
|                                                  |                                                              |                                                              |
|                                                  |                                                              |                                                              |
|                                                  |                                                              |                                                              |



### Problems

#### sqlite3.OperationalError: no such table: log

Solution: it maybe results from absence of environment variable 'FERNET_KEY', you need to go in the file system of the container (as described in the table above), then  `export FERNET_KEY='fewoffeoj_ofoewfeo.....' `

#### Error response from daemon: driver failed programming external connectivity on endpoint

Solution: simply restart docker

####  ERROR: for webapplication2 Cannot create container for service webapplication2: D: drive is not shared. 

Solution: Just right-click docker-app icon, click 'settings', then tick share dive for D:

### To be continued ...