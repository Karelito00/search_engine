import VueRouter from "vue-router";
import Home from "@/components/Home";
import Search from "@/components/Search";
import Vue from "vue";

Vue.use(VueRouter);

const routes = [
  {
    path: "/",
    name: "home",
    component: Home,
  },
  {
    path: "/search",
    name: "search",
    component: Search,
  },
  {
    path: "*",
    redirect: { name: "home" },
  },
];

const router = new VueRouter({
  mode: "history",
  routes,
});

export default router;
