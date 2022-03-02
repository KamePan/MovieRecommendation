import { Layout, Menu, Breadcrumb } from 'antd';
import './index.css';

const { Header, Content, Footer } = Layout;


export default function (props) {
    const { children } = props;
    return (
        <Layout className="layout" style={{ display: 'flex', minHeight: '100vh' }}>
            <Header>
                <div className="logo" />
                <Menu theme="dark" mode="horizontal" defaultSelectedKeys={['2']}>
                    {new Array(3).fill(null).map((_, index) => {
                        const key = index + 1;
                        return <Menu.Item key={key}>{`nav ${key}`}</Menu.Item>;
                    })}
                </Menu>
            </Header>
            <Content style={{ padding: '0 50px', flex: 1, display: 'flex', flexDirection: 'column' }}>
                <div className="site-layout-content">{children}</div>
            </Content>
            <Footer style={{ textAlign: 'center' }}>Ant Design ©2018 Created by Ant UED</Footer>
        </Layout>
    )
}