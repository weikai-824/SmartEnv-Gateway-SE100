'''健康检查，查询服务是否活着'''
from fastapi import APIRouter
from app.config.settings import Settings
#1.创建路由分组对象
router=APIRouter(prefix='/health',tags=['health'])
#2.注册GET接口
@router.get('')
def health_check():
    return {
        'status':'OK',
        'project':Settings.project_name,
        'env':Settings.env
    }